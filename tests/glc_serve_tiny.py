"""A tiny, randomly initialised Qwen3.5 VL checkpoint for CPU tests of glc_serve.

Built entirely offline: the same architecture class as Qwen3.8-27B
(``Qwen3_5ForConditionalGeneration``: Gated-DeltaNet + full-attention hybrid
text trunk, a vision tower, an untied ``lm_head``) plus an ``mtp.*`` head in
the checkpoint, a byte-level BPE tokenizer with the Qwen special tokens, a
chat template with image placeholders and tools, and the image/video
processor configs.  Dimensions are multiples of 64 so the text linears are
eligible for the TBE fragment layout.
"""
from __future__ import annotations

import json
from pathlib import Path

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
           "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
           "<think>", "</think>"]

CHAT_TEMPLATE = (
    "{%- if tools %}<|im_start|>system\nTools: {{ tools | tojson }}<|im_end|>\n{%- endif %}"
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n"
    "{%- if m.content is string %}{{ m.content }}"
    "{%- elif m.content %}{%- for p in m.content %}"
    "{%- if p.type == 'image' or 'image' in p %}<|vision_start|><|image_pad|><|vision_end|>"
    "{%- elif p.type == 'text' %}{{ p.text }}{%- endif %}{%- endfor %}{%- endif %}"
    "<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
)

CORPUS = [
    "The capital of France is Paris, and the capital of Japan is Tokyo.",
    "Water boils at a temperature of one hundred degrees.",
    "In 1969, humans first walked on the moon.",
    "The chemical symbol for gold is Au.",
    "def add(a, b): return a + b",
    "Describe the shapes and colours in this image, and read any text.",
    "Which bar in this chart is the tallest? Answer with its position.",
] * 20


def build_tokenizer(out: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(CORPUS, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL[1:],
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(str(out))
    return fast


def build_checkpoint(out: Path, *, seed: int = 0, with_mtp: bool = True, vocab: int = 512):
    """Write a complete tiny checkpoint dir (weights + tokenizer + processor)."""
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer(out)
    ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIAL}
    text = dict(
        vocab_size=vocab, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_conv_kernel_dim=4, full_attention_interval=4,
        max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=False,
        attn_output_gate=True, mtp_num_hidden_layers=1, eos_token_id=ids["<|im_end|>"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [2, 3, 3],
                         "mrope_interleaved": True},
    )
    vision = dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=2,
                  out_hidden_size=128, patch_size=16, spatial_merge_size=2,
                  temporal_patch_size=2, num_position_embeddings=64, in_channels=3,
                  deepstack_visual_indexes=[])
    cfg = Qwen3_5Config(text_config=text, vision_config=vision,
                        image_token_id=ids["<|image_pad|>"],
                        video_token_id=ids["<|video_pad|>"],
                        vision_start_token_id=ids["<|vision_start|>"],
                        vision_end_token_id=ids["<|vision_end|>"],
                        tie_word_embeddings=False)
    cfg.architectures = ["Qwen3_5ForConditionalGeneration"]
    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    # random init leaves RMSNorm weights at 0 (scale 1+w); perturb every tensor so
    # a wrong placement cannot hide behind an all-zeros coincidence
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p.float()).to(p.dtype) * 0.02)
    model.generation_config.eos_token_id = [ids["<|im_end|>"], ids["<|endoftext|>"]]
    model.generation_config.pad_token_id = ids["<|endoftext|>"]
    model.save_pretrained(str(out), safe_serialization=True)
    if with_mtp:
        from glc_serve.mtp import QwenMTP

        mtp = QwenMTP(cfg.text_config).to(torch.bfloat16)
        with torch.no_grad():
            for p in mtp.parameters():
                p.copy_((torch.randn_like(p.float()) * 0.05).to(p.dtype))
        state = {f"mtp.{k}": v.contiguous() for k, v in mtp.state_dict().items()}
        save_file(state, str(out / "model-mtp.safetensors"))
        idx_path = out / "model.safetensors.index.json"
        if idx_path.is_file():
            idx = json.loads(idx_path.read_text())
        else:
            names = {}
            from safetensors import safe_open

            with safe_open(str(out / "model.safetensors"), framework="pt") as h:
                for k in h.keys():
                    names[k] = "model.safetensors"
            idx = {"metadata": {}, "weight_map": names}
        for k in state:
            idx["weight_map"][k] = "model-mtp.safetensors"
        idx_path.write_text(json.dumps(idx, indent=1))
    build_processor(out, tok)
    return cfg


def build_processor(out: Path, tok):
    pre = {
        "size": {"longest_edge": 64 * 64 * 4, "shortest_edge": 32 * 32},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessor",
    }
    (out / "preprocessor_config.json").write_text(json.dumps(pre, indent=1))
    vid = dict(pre, video_processor_type="Qwen3VLVideoProcessor")
    vid.pop("image_processor_type", None)
    (out / "video_preprocessor_config.json").write_text(json.dumps(vid, indent=1))
    (out / "chat_template.jinja").write_text(CHAT_TEMPLATE)
