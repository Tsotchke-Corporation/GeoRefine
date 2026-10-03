"""What actually limits concurrency: per-user bytes, and the context at which KV overtakes state.

The reflex answer to "how many users fit" is to compress weights.  For this hybrid that is the
wrong lever over part of the range, and the arithmetic says exactly where the line is:

* a **full-attention** layer's KV grows with context -- ``2 (K and V) * n_kv * head_dim * 2 B``
  per token per layer, over the 16 attention layers;
* a **Gated-DeltaNet** layer's recurrent state does **not** grow with context -- it is a fixed
  ``n_heads * d_k * d_v`` fp32 matrix plus a conv ring, per slot, over the 48 GDN layers.

So there is a crossover context length.  Below it the fixed GDN state is the binding term and no
KV technique helps at all; above it KV dominates and paging, prefix sharing and KV coding are
the levers.  This module computes that crossover and the per-user budget from the engine's own
tensor shapes -- no model forward, no GPU -- so ``--pages``, ``--max-users`` and ``--max-ctx``
can be chosen from arithmetic instead of from a guess, and so a concurrency claim can be
checked against the geometry that produces it.

    python -m release.glc_serve.bidec_capacity --from-receipt receipt.json --vram-gib 80
    python -m release.glc_serve.bidec_capacity --n-att 16 --n-gdn 48 --n-kv 4 --head-dim 128 \\
        --gdn-heads 32 --gdn-dk 128 --gdn-dv 128 --vram-gib 80 --weight-gib 20
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PAGE = 256
GIB = 1 << 30


@dataclass
class Geometry:
    """Only the shapes that set per-user cost."""
    n_att: int                      # full-attention layers (KV grows with context)
    n_gdn: int                      # Gated-DeltaNet layers (state is context-independent)
    n_kv: int                       # KV heads per attention layer (GQA group count)
    head_dim: int                   # per-head dimension
    kv_dtype_bytes: int = 2         # bf16/fp16 KV
    gdn_heads: int = 32             # recurrent-state heads
    gdn_dk: int = 128
    gdn_dv: int = 128
    gdn_state_dtype_bytes: int = 4  # fp32 recurrent state
    conv_width: int = 16
    conv_channels: int = 0          # 0 -> estimated as gdn_heads * gdn_dk
    conv_dtype_bytes: int = 2
    page_tokens: int = PAGE

    # ---------------------------------------------------------------- per-user terms
    def kv_bytes_per_token(self) -> int:
        """KV bytes added by ONE more token of context, across the attention layers."""
        return self.n_att * 2 * self.n_kv * self.head_dim * self.kv_dtype_bytes

    def kv_bytes(self, ctx: int) -> int:
        """KV actually allocated for a context of `ctx` tokens: whole pages, as the engine does."""
        pages = (max(0, int(ctx)) + self.page_tokens - 1) // self.page_tokens + 1
        return pages * self.page_tokens * self.kv_bytes_per_token()

    def gdn_state_bytes(self) -> int:
        """Per-slot recurrent state + conv history.  Independent of context length."""
        ch = self.conv_channels or (self.gdn_heads * self.gdn_dk)
        recur = self.n_gdn * self.gdn_heads * self.gdn_dk * self.gdn_dv * self.gdn_state_dtype_bytes
        conv = self.n_gdn * self.conv_width * ch * self.conv_dtype_bytes
        return recur + conv

    def bytes_per_user(self, ctx: int, *, state_ring: int = 1) -> int:
        return self.kv_bytes(ctx) + state_ring * self.gdn_state_bytes()

    # ---------------------------------------------------------------- the crossover
    def crossover_tokens(self, *, state_ring: int = 1) -> int:
        """Context length at which a user's KV first exceeds that user's GDN state.

        Below this, concurrency is set by a fixed per-slot tax that no KV technique touches;
        above it, KV is the lever.  Returned in tokens, rounded up to a page.
        """
        per_tok = self.kv_bytes_per_token()
        if per_tok <= 0:
            return 0
        raw = (state_ring * self.gdn_state_bytes() + per_tok - 1) // per_tok
        return int(((raw + self.page_tokens - 1) // self.page_tokens) * self.page_tokens)

    def max_users(self, ctx: int, budget_bytes: int, *, state_ring: int = 1) -> int:
        per = self.bytes_per_user(ctx, state_ring=state_ring)
        return 0 if per <= 0 else max(0, int(budget_bytes // per))

    def pages_for(self, ctx: int, users: int) -> int:
        """The --pages value that holds `users` concurrent requests at this context."""
        per = (max(0, int(ctx)) + self.page_tokens - 1) // self.page_tokens + 1
        return per * max(0, int(users))

    # ---------------------------------------------------------------- report
    def plan(self, *, ctxs=(1024, 2048, 4096, 8192, 16384, 32768), vram_bytes: int = 0,
             weight_bytes: int = 0, state_ring: int = 1) -> Dict[str, Any]:
        budget = max(0, vram_bytes - weight_bytes)
        rows = []
        for c in ctxs:
            kv, st = self.kv_bytes(c), state_ring * self.gdn_state_bytes()
            rows.append({"ctx": c, "kv_mib": round(kv / (1 << 20), 1),
                         "gdn_state_mib": round(st / (1 << 20), 1),
                         "bytes_per_user_mib": round((kv + st) / (1 << 20), 1),
                         "kv_share": round(kv / (kv + st), 4) if kv + st else None,
                         "binding_term": "kv" if kv > st else "gdn_state",
                         "max_users_in_budget": self.max_users(c, budget, state_ring=state_ring)
                         if budget else None,
                         "pages_for_that": self.pages_for(c, self.max_users(c, budget, state_ring=state_ring))
                         if budget else None})
        x = self.crossover_tokens(state_ring=state_ring)
        half = Geometry(**{**asdict(self), "gdn_state_dtype_bytes": 2})
        ring_alt = {
            "current_state_dtype_bytes": self.gdn_state_dtype_bytes,
            "fp16_ring_state_bytes_per_slot": half.gdn_state_bytes(),
            "fp16_ring_saving_bytes_per_slot": self.gdn_state_bytes() - half.gdn_state_bytes(),
            "fp16_ring_crossover_tokens": half.crossover_tokens(state_ring=state_ring),
            "status": "PROJECTED from the shapes; measured behaviour-lossless on Qwen3.5-2B "
                      "(greedy 1.000000 at 8k and over 64 round-trips, top-5 0.995, rare-tail "
                      "gap <= 0.17 pt), NOT yet gated on the served lane and NOT yet supported "
                      "by b_gdn_recur, which types the ring as float*"}
        return {
            "geometry": asdict(self),
            "kv_bytes_per_token": self.kv_bytes_per_token(),
            "gdn_state_bytes_per_slot": self.gdn_state_bytes(),
            "state_ring": state_ring,
            "crossover_tokens": x,
            "crossover_note":
                f"below ~{x} tokens of context the fixed per-slot GDN state is the binding term, "
                f"so KV paging / sharing / coding cannot raise the user count there; above it KV "
                f"dominates and those are exactly the levers that do",
            "vram_bytes": vram_bytes or None, "weight_bytes": weight_bytes or None,
            "kv_and_state_budget_bytes": budget or None,
            "per_context": rows, "gdn_ring_alternative": ring_alt,
            "levers": {
                "below_crossover": ["per-slot state precision where it is bit-identical",
                                    "state snapshot reuse across users sharing a prefix",
                                    "fewer resident slots, deeper queue"],
                "above_crossover": ["paged KV from a shared pool (shipped)",
                                    "shared-prefix KV reuse (landing)",
                                    "lossless KV coding on the attention layers",
                                    "latent/cross-layer KV on the 16 attention layers"]},
        }


def from_receipt(obj: Dict[str, Any]) -> Optional[Geometry]:
    """Build a Geometry from a ``/v1/receipt`` body, when it carries the shapes."""
    g = obj.get("geometry") or (obj.get("meta") or {}).get("geometry")
    if not g:
        return None
    known = Geometry.__dataclass_fields__
    return Geometry(**{k: v for k, v in g.items() if k in known})


def geometry_of(bd: Any) -> Geometry:
    """Read the shapes off a live BatchDecoder (or anything exposing the same fields)."""
    fd = getattr(bd, "fd", None)
    n_att = getattr(fd, "n_att", None) if fd is not None else None
    n_gdn = getattr(fd, "n_gdn", None) if fd is not None else None
    if n_att is None or n_gdn is None:
        raise ValueError("decoder does not expose n_att / n_gdn")
    # CLASSIFICATION: production_fallback.  n_att / n_gdn are refused above rather than guessed,
    # because they set the whole arithmetic.  The recurrent-state and conv shapes fall back to the
    # Qwen3.5/3.8 hybrid's defaults when the buffers are not present (a decoder that has not
    # allocated them yet, or the CPU reference), and `fallbacks_used` records it so a plan built
    # on a default is never mistaken for one read off real buffers.
    st = getattr(bd, "state", None)
    used = []
    if st is not None and st.dim() >= 5:
        gh, dk, dv = st.shape[2], st.shape[3], st.shape[4]
    else:
        gh, dk, dv = 32, 128, 128
        used.append("gdn_heads/dk/dv")
    hist = getattr(bd, "hist", None)
    if hist is not None and hist.dim() >= 4:
        cw, ch = hist.shape[2], hist.shape[3]
    else:
        cw, ch = 16, 0
        used.append("conv_width/channels")
    g = Geometry(n_att=int(n_att), n_gdn=int(n_gdn), n_kv=int(getattr(bd, "NKV", 4)),
                 head_dim=int(getattr(bd, "head_dim", 256)),
                 gdn_heads=int(gh), gdn_dk=int(dk), gdn_dv=int(dv),
                 conv_width=int(cw), conv_channels=int(ch), page_tokens=PAGE)
    g.fallbacks_used = tuple(used)                       # noqa: attribute set for provenance
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-receipt", help="a saved GET /v1/receipt body")
    ap.add_argument("--n-att", type=int, default=16)
    ap.add_argument("--n-gdn", type=int, default=48)
    ap.add_argument("--n-kv", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--gdn-heads", type=int, default=32)
    ap.add_argument("--gdn-dk", type=int, default=128)
    ap.add_argument("--gdn-dv", type=int, default=128)
    ap.add_argument("--gdn-ring-dtype", choices=("fp32", "fp16"), default="fp32",
                    help="storage dtype of the GDN recurrent ring for this plan")
    ap.add_argument("--state-ring", type=int, default=1, help="R: spec depth k needs R = k + 1")
    ap.add_argument("--vram-gib", type=float, default=0.0)
    ap.add_argument("--weight-gib", type=float, default=0.0)
    ap.add_argument("--ctxs", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--out")
    a = ap.parse_args()

    g = None
    if a.from_receipt:
        g = from_receipt(json.loads(Path(a.from_receipt).read_text()))
        if g is None:
            raise SystemExit("that receipt carries no geometry block; pass the shapes explicitly")
    if g is None:
        g = Geometry(n_att=a.n_att, n_gdn=a.n_gdn, n_kv=a.n_kv, head_dim=a.head_dim,
                     gdn_heads=a.gdn_heads, gdn_dk=a.gdn_dk, gdn_dv=a.gdn_dv,
                     gdn_state_dtype_bytes=4 if a.gdn_ring_dtype == "fp32" else 2)
    rep = g.plan(ctxs=tuple(int(x) for x in a.ctxs.split(",") if x.strip()),
                 vram_bytes=int(a.vram_gib * GIB), weight_bytes=int(a.weight_gib * GIB),
                 state_ring=a.state_ring)
    text = json.dumps(rep, indent=2)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
