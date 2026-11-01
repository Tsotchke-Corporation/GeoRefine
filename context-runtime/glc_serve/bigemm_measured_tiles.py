"""Exact-shape BI-GEMM tile measurements from the pinned operator receipt."""
from __future__ import annotations

import hashlib
import json

MEASURED_ARCH = (12, 0, 188)
MEASURED_SOURCE_REVISION = "65c6a24207266341094c1f8d691e8a9841ee657a"
MEASURED_GPU_UUID = "GPU-5c0cbdf0-315b-d357-9833-c5c7612cc00c"
RECEIPT_SHA256 = "8b577ca2b5f90c7447600746a2f201e3b5c7d48eb206491de0e51088f5941d07"
RECEIPT_STATUS = "PASS_OPERATOR_ONLY"
# Dispatch-selection margin only; it does not alter any release or certification gate.
MIN_SPEEDUP_TO_SELECT_128 = 1.05

# Values are (tile64_ms, tile128_ms); M keys are measured rows only.
TIMINGS_MS = {
    (5120, 17408): {
        64: {
            'bf16': (0.13206399977207184, 0.13939200341701508),
            'tbe': (0.14300799369812012, 0.19203199446201324),
        },
        128: {
            'bf16': (0.23756800591945648, 0.1578879952430725),
            'tbe': (0.26844799518585205, 0.21641600131988525),
        },
        256: {
            'bf16': (0.4665600061416626, 0.3250559866428375),
            'tbe': (0.49324798583984375, 0.4193280041217804),
        },
        512: {
            'bf16': (0.9579200148582458, 0.5893440246582031),
            'tbe': (0.9399999976158142, 0.7447999715805054),
        },
    },
    (14336, 5120): {
        64: {
            'bf16': (0.12544000148773193, 0.15568000078201294),
            'tbe': (0.16179199516773224, 0.21321600675582886),
        },
        128: {
            'bf16': (0.1910720020532608, 0.17820799350738525),
            'tbe': (0.24380800127983093, 0.23820799589157104),
        },
        256: {
            'bf16': (0.3436160087585449, 0.26611199975013733),
            'tbe': (0.3829439878463745, 0.3543359935283661),
        },
        512: {
            'bf16': (0.7287359833717346, 0.4503999948501587),
            'tbe': (0.7187520265579224, 0.5904639959335327),
        },
    },
    (16480, 5120): {
        64: {
            'bf16': (0.13203200697898865, 0.15772800147533417),
            'tbe': (0.1621759980916977, 0.2146880030632019),
        },
        128: {
            'bf16': (0.21024000644683838, 0.1817599982023239),
            'tbe': (0.2526400089263916, 0.24246400594711304),
        },
        256: {
            'bf16': (0.4102399945259094, 0.274944007396698),
            'tbe': (0.45187199115753174, 0.36259201169013977),
        },
        512: {
            'bf16': (0.8863360285758972, 0.5421440005302429),
            'tbe': (0.8504639863967896, 0.7132160067558289),
        },
    },
    (34816, 5120): {
        64: {
            'bf16': (0.26284798979759216, 0.258432000875473),
            'tbe': (0.308351993560791, 0.33372798562049866),
        },
        128: {
            'bf16': (0.48841598629951477, 0.2812480032444),
            'tbe': (0.5626879930496216, 0.38284799456596375),
        },
        256: {
            'bf16': (0.9635519981384277, 0.5481280088424683),
            'tbe': (1.1007360219955444, 0.7411199808120728),
        },
        512: {
            'bf16': (1.9130239486694336, 1.052832007408142),
            'tbe': (2.159775972366333, 1.456223964691162),
        },
    },
    (248320, 5120): {
        64: {
            'bf16': (1.69651198387146, 1.732319951057434),
            'tbe': (1.9903039932250977, 2.1832640171051025),
        },
        128: {
            'bf16': (3.424704074859619, 1.8151040077209473),
            'tbe': (3.859839916229248, 2.460319995880127),
        },
        256: {
            'bf16': (6.898303985595703, 3.671488046646118),
            'tbe': (7.465472221374512, 4.850175857543945),
        },
        512: {
            'bf16': (13.846528053283691, 7.34771203994751),
            'tbe': (14.800383567810059, 9.587136268615723),
        },
    },
}


def policy_applies(fmt, cap, arch):
    """Whether this process is on the explicitly enabled measured policy path."""
    if int(cap) < 128 or fmt not in ("bf16", "tbe"):
        return None
    try:
        device_arch = tuple(int(part) for part in arch)
    except (TypeError, ValueError):
        return False
    return device_arch == MEASURED_ARCH


def select_measured_tile(fmt, n, k, m, cap, arch, *, explicit_tile=None):
    """Choose only from exact measured shape/row/format/arch data, else return None."""
    if explicit_tile is not None or not policy_applies(fmt, cap, arch):
        return None
    pair = TIMINGS_MS.get((int(n), int(k)), {}).get(int(m), {}).get(fmt)
    if pair is None:
        return None
    if int(m) <= 64:
        return 64
    speedup = pair[0] / pair[1]
    return 128 if speedup > MIN_SPEEDUP_TO_SELECT_128 else 64


def policy_identity():
    serializable = {
        f"{n}x{k}": {
            str(m): {fmt: list(pair) for fmt, pair in formats.items()}
            for m, formats in rows.items()
        }
        for (n, k), rows in TIMINGS_MS.items()
    }
    data_sha = hashlib.sha256(json.dumps(serializable, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    return {"name": "measured-bi-gemm-tile-v1", "data_sha256": data_sha,
            "receipt_sha256": RECEIPT_SHA256,
            "measured_source_revision": MEASURED_SOURCE_REVISION,
            "measured_gpu_uuid": MEASURED_GPU_UUID, "arch": list(MEASURED_ARCH),
            "minimum_speedup_to_select_128": MIN_SPEEDUP_TO_SELECT_128,
            "selection_margin_scope": "tile dispatch only; not a release gate",
            "M_le_64_tile": 64,
            "scope": "exact shape, M, format and architecture only"}
