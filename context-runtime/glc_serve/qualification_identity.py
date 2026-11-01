"""Bind a qualification endpoint to actual installed code and local artifacts."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import math
import struct
from pathlib import Path,PurePosixPath
import zipfile

PACKAGES=('glc_serve','glc_loader','georefine_context_runtime')
MANIFEST='georefine_context_runtime/SOURCE_MANIFEST.json'


def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def package_manifest_path(package):
    root=Path(package)
    for name in ('manifest.json','model.safetensors.index.json'):
        path=root/name
        if path.is_file():return path
    raise ValueError('package requires a compressed manifest or dense safetensors index')


TBE21_SOURCE_NAMES={"tbe21_v2_backend.py", "bi_tbe21_v2_csrc/bi_torch.cpp",
    "bi_tbe21_v2_csrc/bi_tbe21_v2_api.h", "bi_tbe21_v2_csrc/bi_gemm_v2.cu",
    "bi_tbe21_v2_csrc/bi_gemm.cuh", "bi_tbe21_v2_csrc/pair_vote.cuh", "bi_tbe21_v2_csrc/bi_api.h",
    "bi_tbe21_v2_csrc/miv_tbe_kernels.cuh", "bi_tbe21_v2_csrc/miv_tbe.h",
    "bi_tbe21_v2_csrc/tier_prefix_literal.cuh", "bi_tbe21_v2_csrc/tier_prefix_cursor.cuh"}
TBE21_KERNEL_VARIANT_ENV = "BI_TBE21_KERNEL_VARIANT"
TBE21_TIER_PREFIX_ENV = ("BI_T21_TIER_PREFIX", "BI_T21_TPF_PREFETCH",
                         "BI_T21_TPF_WG1", "BI_T21_TPF_PREFETCH_WG1")
TBE21_TIER_PREFIX_IDENTITY = ("tier_prefix", "tier_prefetch", "tier_prefix_wg1",
                              "tier_prefetch_wg1")


def _tier_profile_suffix(source):
    from glc_serve import tbe21_tier_profile as tier
    return tier.suffix(tier.validate_source(source))


def validate_tbe21_tier_prefix_disabled(source=None, environ=None, identity=None):
    """Compatibility entry: reject unbound tier modes, validate explicit profiles."""
    from glc_serve import tbe21_tier_profile as tier
    value = {} if source is None else source
    if not isinstance(value, dict):
        raise ValueError("tier-prefix source binding must be an object")
    profile = tier.validate_source(value)
    if environ is not None:
        tier.validate_environment(value, environ)
    if identity is not None:
        tier.validate_identity(identity, profile)
        if profile is not None and identity.get('row_geometry', False) != value.get('row_geometry', False):
            raise ValueError('tier-prefix row geometry identity differs from source binding')
    return profile


CPU_ENCODER_ENV = 'GEOREFINE_TBE21_NATIVE_PACK'
CPU_ENCODER_PATHS = {name: 'glc_loader/' + name for name in
    ('codec_v21.py', 'codec_v21_native.py', 'codec_v21_native.cpp')}
CPU_ENCODER_FLAGS = {system: ['-O3', '-std=c++17', '-fPIC', link]
    for system, link in (('Linux', '-shared'), ('Darwin', '-dynamiclib'))}


def _encoder_sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_cpu_encoder_binding(source):
    if 'cpu_encoder' not in source:
        return None
    value = source['cpu_encoder']
    if source.get('kind') != 'context_v21_transcode':
        raise ValueError('cpu_encoder is supported only for context_v21_transcode')
    if (not isinstance(value, dict) or set(value) != {'mode', 'abi', 'threads', 'source_sha256', 'flags'}
            or value.get('mode') != 'native_tbe21' or type(value.get('abi')) is not int or value['abi'] != 1
            or type(value.get('threads')) is not int or value['threads'] != 1
            or not isinstance(value.get('source_sha256'), dict)
            or set(value['source_sha256']) != set(CPU_ENCODER_PATHS)
            or any(not _encoder_sha(h) for h in value['source_sha256'].values())
            or value.get('flags') != CPU_ENCODER_FLAGS):
        raise ValueError('cpu_encoder requires exact native_tbe21 source/ABI/threads/flags identity')
    return {**value, 'source_sha256': dict(value['source_sha256']),
            'flags': {key: list(flags) for key, flags in value['flags'].items()}}


def cpu_encoder_from_manifest(manifest):
    rows = manifest.get('sources')
    if not isinstance(rows, list):
        raise ValueError('cpu_encoder source manifest is malformed')
    found = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('cpu_encoder source manifest row is malformed')
        for key, path in CPU_ENCODER_PATHS.items():
            if row.get('installed_path') == path:
                if key in found:
                    raise ValueError('duplicate cpu_encoder source manifest path')
                found[key] = row.get('installed_sha256')
    return validate_cpu_encoder_binding({'kind': 'context_v21_transcode', 'cpu_encoder': {
        'mode': 'native_tbe21', 'abi': 1, 'threads': 1, 'source_sha256': found, 'flags': CPU_ENCODER_FLAGS}})


def validate_cpu_encoder_environment(source, environ):
    expected = validate_cpu_encoder_binding(source)
    value = environ.get(CPU_ENCODER_ENV, '0')
    if value not in ('0', '1'):
        raise ValueError('GEOREFINE_TBE21_NATIVE_PACK must be exactly 0 or 1')
    if expected is None and value == '1':
        raise ValueError('native TBE21 encoder is not bound by qualification protocol')
    if expected is not None and value != '1':
        raise ValueError('cpu_encoder requires explicit GEOREFINE_TBE21_NATIVE_PACK=1')
    return expected


def validate_cpu_encoder_record(expected, record):
    fields = {'recipe', 'key', 'binary_path', 'binary_sha256', 'loaded_binary_sha256',
              'compiled_cpp_sha256', 'build_record_sha256'}
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError('cpu_encoder native producer receipt is malformed')
    recipe = record['recipe']
    recipe_fields = {'abi', 'source_sha256', 'compiler', 'compiler_version', 'compiler_stat',
                     'flags', 'system', 'machine', 'pointer_bytes', 'threads'}
    if not isinstance(recipe, dict) or set(recipe) != recipe_fields:
        raise ValueError('cpu_encoder native recipe is malformed')
    system = recipe.get('system')
    cpp_sha = expected['source_sha256']['codec_v21_native.cpp']
    if (system not in CPU_ENCODER_FLAGS or recipe.get('source_sha256') != expected['source_sha256']
            or type(recipe.get('abi')) is not int or recipe['abi'] != expected['abi']
            or type(recipe.get('threads')) is not int or recipe['threads'] != expected['threads']
            or recipe.get('flags') != expected['flags'][system] + [f'-DT21_SOURCE_SHA256="{cpp_sha}"']
            or record.get('compiled_cpp_sha256') != cpp_sha):
        raise ValueError('cpu_encoder native source/ABI/threads/flags/getter differs from protocol')
    if (not isinstance(recipe.get('compiler'), str) or not recipe['compiler'].startswith('/')
            or not isinstance(recipe.get('compiler_version'), str) or not recipe['compiler_version'].strip()
            or not isinstance(recipe.get('machine'), str) or not recipe['machine'].strip()
            or type(recipe.get('pointer_bytes')) is not int or recipe['pointer_bytes'] != 8
            or not isinstance(recipe.get('compiler_stat'), list) or len(recipe['compiler_stat']) != 4
            or any(type(n) is not int or n < 0 for n in recipe['compiler_stat'])
            or not isinstance(record.get('binary_path'), str) or not record['binary_path'].startswith('/')
            or any(not _encoder_sha(record.get(key)) for key in
                   ('key', 'binary_sha256', 'loaded_binary_sha256', 'build_record_sha256'))
            or record['binary_sha256'] != record['loaded_binary_sha256']):
        raise ValueError('cpu_encoder native build/binary identity is invalid')
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    build_record = {key: record[key] for key in ('recipe', 'key', 'binary_path', 'binary_sha256')}
    build_sha = hashlib.sha256((json.dumps(build_record, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    if record['key'] != key or record['build_record_sha256'] != build_sha:
        raise ValueError('cpu_encoder producer build record hash mismatch')
    return record['build_record_sha256']


def validate_cpu_encoder_receipt(source, load):
    expected = validate_cpu_encoder_binding(source)
    markers = {}
    def visit(value, path=()):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'runtime_encoder':
                    markers[path + (key,)] = item
                else:
                    visit(item, path + (key,))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, path + (i,))
    visit(load)
    if expected is None:
        if markers:
            raise ValueError('native TBE21 encoder is not bound by qualification protocol')
        return None
    tbe21 = load.get('tbe21') if isinstance(load, dict) else None
    transcode = tbe21.get('context_transcode') if isinstance(tbe21, dict) else None
    rows = transcode.get('per_tensor') if isinstance(transcode, dict) else None
    counts = tbe21.get('counts') if isinstance(tbe21, dict) else None
    coded_count = counts.get('tbe2_linear') if isinstance(counts, dict) else None
    if (type(coded_count) is not int or coded_count <= 0
            or not isinstance(rows, list) or not rows or type(transcode.get('tensors')) is not int
            or transcode['tensors'] != len(rows) or len(rows) != coded_count
            or any(not isinstance(row, dict) or not isinstance(row.get('name'), str)
                   or not row['name'] for row in rows)):
        raise ValueError('cpu_encoder requires actual per-tensor native producer receipts')
    if len({row['name'] for row in rows}) != len(rows):
        raise ValueError('cpu_encoder per-tensor receipt names are duplicated')
    wanted = {('tbe21', 'context_transcode', 'per_tensor', i, 'runtime_encoder') for i in range(len(rows))}
    if not wanted.issubset(markers):
        raise ValueError('cpu_encoder per-tensor native producer markers are missing')
    builds = {validate_cpu_encoder_record(expected, record) for record in markers.values()}
    if len(builds) != 1:
        raise ValueError('cpu_encoder native producer build identity differs across tensors/workers')
    return next(iter(builds))


def validate_tbe21_tile128(source):
    selected = source.get("tile128", False)
    if "tile128" in source and selected is not True:
        raise ValueError("tile128 must be true when present")
    if selected and (validate_tbe21_kernel_variant(source.get("kernel_variant", 2)) != 1
                     or not validate_tbe21_packed_scan2(source)):
        raise ValueError("tile128 requires TBE21 kernel_variant 1 and packed_scan2")
    return selected


def validate_tbe21_tile128_environment(source, env):
    expected = validate_tbe21_tile128(source)
    raw = env.get("BI_T21_TILE128", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_TILE128 must be exactly 0 or 1")
    if raw == "1" and (env.get("BI_TBE21_KERNEL_VARIANT", "2") != "1"
                       or env.get("BI_T21_PACKSCAN2", "0") != "1"):
        raise ValueError("tile128 environment requires kernel_variant 1 and packed_scan2")
    if (raw == "1") != expected:
        raise ValueError("TBE21 tile128 environment differs from binding")
    return expected


def validate_tbe21_tile_metadata(metadata, source):
    selected = validate_tbe21_tile128(source)
    actual = metadata.get("tbe21_tile_policy")
    if selected:
        if (actual != {"rows_le_64": 64, "rows_gt_64": 128}
                or not isinstance(actual, dict) or any(type(v) is not int for v in actual.values())
                or metadata.get("gemm_tile_scope") != "legacy formats; TBE21 uses tbe21_tile_policy"):
            raise ValueError("TBE21 tile128 runtime policy metadata missing or malformed")
    elif actual is not None or "gemm_tile_scope" in metadata:
        raise ValueError("unbound TBE21 tile128 runtime policy metadata")


def validate_tbe21_tile_identity(identity, selected, *, descriptor=False, row_geometry=False, exec_layout=False, scan4=False, wide128=False, pipe128=False, packed_scan4=False):
    if type(selected) is not bool or not isinstance(identity, dict):
        raise ValueError("TBE21 tile128 identity malformed")
    if type(identity.get("tile128", False)) is not bool or identity.get("tile128", False) != selected:
        raise ValueError("TBE21 tile128 identity differs from binding")
    if (type(row_geometry) is not bool or row_geometry and not selected
            or type(identity.get("row_geometry", False)) is not bool
            or identity.get("row_geometry", False) != row_geometry):
        raise ValueError("TBE21 row geometry identity differs from binding")
    if (type(exec_layout) is not bool or exec_layout and (not selected or row_geometry)
            or type(identity.get("exec_layout", False)) is not bool
            or identity.get("exec_layout", False) != exec_layout):
        raise ValueError("TBE21 execution layout identity differs from binding")
    if (type(scan4) is not bool or type(identity.get("scan4", False)) is not bool
            or identity.get("scan4", False) != scan4):
        raise ValueError("TBE21 scan4 identity differs from binding")
    if (type(wide128) is not bool or wide128 and (not selected or scan4)
            or type(identity.get("wide128", False)) is not bool
            or identity.get("wide128", False) != wide128):
        raise ValueError("TBE21 wide128 identity differs from binding")
    if (type(pipe128) is not bool or pipe128 and (not selected or scan4 or not wide128)
            or type(identity.get("pipe128", False)) is not bool
            or identity.get("pipe128", False) != pipe128):
        raise ValueError("TBE21 pipe128 identity differs from binding")
    if (type(packed_scan4) is not bool or packed_scan4 and (scan4 or wide128 or pipe128 or row_geometry)
            or "packed_scan4" in identity and type(identity.get("packed_scan4")) is not bool
            or identity.get("packed_scan4", False) != packed_scan4):
        raise ValueError("TBE21 packed_scan4 identity differs from binding")
    if selected:
        policy = identity.get("tile_policy")
        if ("tile" in identity or not isinstance(policy, dict)
                or policy != {"rows_le_64": 64, "rows_gt_64": 128}
                or any(type(v) is not int for v in policy.values())):
            raise ValueError("TBE21 tile128 policy missing or malformed")
        if descriptor:
            counts = identity.get("tile_dispatch_counts")
            if (type(identity.get("tile_cap")) is not int or identity["tile_cap"] != 128
                    or not isinstance(counts, dict) or set(counts) != {"64", "128"}
                    or any(type(v) is not int or v < 0 for v in counts.values())
                    or type(identity.get("dispatch_calls")) is not int
                    or sum(counts.values()) != identity["dispatch_calls"]):
                raise ValueError("TBE21 tile128 successful dispatch counts malformed")
    elif (type(identity.get("tile")) is not int or identity["tile"] != 64
          or "tile_policy" in identity or "tile_dispatch_counts" in identity
          or descriptor and (type(identity.get("tile_cap")) is not int or identity["tile_cap"] != 64)):
        raise ValueError("TBE21 default tile identity must remain 64")


def validate_tbe21_packed_scan2(source):
    selected = source.get("packed_scan2", False)
    if "packed_scan2" in source and selected is not True:
        raise ValueError("packed_scan2 must be true when present")
    if selected and validate_tbe21_kernel_variant(source.get("kernel_variant", 2)) != 1:
        raise ValueError("packed_scan2 requires TBE21 kernel_variant 1")
    return selected


def validate_tbe21_packed_scan2_environment(source, env):
    expected = validate_tbe21_packed_scan2(source)
    raw = env.get("BI_T21_PACKSCAN2", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_PACKSCAN2 must be exactly 0 or 1")
    if raw == "1" and env.get("BI_TBE21_KERNEL_VARIANT", "2") != "1":
        raise ValueError("packed_scan2 environment requires TBE21 kernel_variant 1")
    if (raw == "1") != expected:
        raise ValueError("TBE21 packed_scan2 environment differs from binding")
    return expected

def validate_tbe21_wide128(source):
    selected = source.get("wide128", False)
    if "wide128" in source and selected is not True:
        raise ValueError("wide128 must be true when present")
    if selected and (not validate_tbe21_tile128(source) or source.get("scan4", False)):
        raise ValueError("wide128 requires schedule 1, packed_scan2, tile128 and SCAN4 off")
    return selected


def validate_tbe21_wide128_environment(source, env):
    expected = validate_tbe21_wide128(source)
    raw = env.get("BI_T21_WIDE128", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_WIDE128 must be exactly 0 or 1")
    if raw == "1" and (env.get("BI_TBE21_V2", "0") != "1"
                       or env.get("BI_TBE21_KERNEL_VARIANT", "2") != "1"
                       or env.get("BI_T21_PACKSCAN2", "0") != "1"
                       or env.get("BI_T21_TILE128", "0") != "1"
                       or env.get("BI_T21_SCAN4", "0") != "0"):
        raise ValueError("wide128 environment requires native schedule 1, paired scan, tile128 and SCAN4 off")
    if (raw == "1") != expected:
        raise ValueError("TBE21 wide128 environment differs from binding")
    return expected


def validate_tbe21_pipe128(source):
    selected = source.get("pipe128", False)
    if "pipe128" in source and selected is not True:
        raise ValueError("pipe128 must be true when present")
    if selected and (not validate_tbe21_tile128(source) or validate_tbe21_scan4(source)
                     or validate_tbe21_async_x(source) or not validate_tbe21_wide128(source)):
        raise ValueError("pipe128 requires schedule 1, packed_scan2, tile128, WIDE128 on and SCAN4/ASYNC_X off")
    return selected


def validate_tbe21_pipe128_environment(source, env):
    expected = validate_tbe21_pipe128(source)
    raw = env.get("BI_T21_PIPE128", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_PIPE128 must be exactly 0 or 1")
    if raw == "1" and (env.get("BI_TBE21_V2", "0") != "1"
                       or env.get("BI_TBE21_KERNEL_VARIANT", "2") != "1"
                       or env.get("BI_T21_PACKSCAN2", "0") != "1"
                       or env.get("BI_T21_TILE128", "0") != "1"
                       or env.get("BI_T21_SCAN4", "0") != "0"
                       or env.get("BI_T21_ASYNC_X", "0") != "0"
                       or env.get("BI_T21_WIDE128", "0") != "1"):
        raise ValueError("pipe128 environment requires native schedule 1, paired scan, tile128, WIDE128 on and SCAN4/ASYNC_X off")
    if (raw == "1") != expected:
        raise ValueError("TBE21 pipe128 environment differs from binding")
    return expected


def validate_tbe21_packed_scan4(source):
    selected = source.get("packed_scan4", False)
    if "packed_scan4" in source and selected is not True:
        raise ValueError("packed_scan4 must be true when present")
    if selected and (validate_tbe21_kernel_variant(source.get("kernel_variant", 2)) != 1
                     or not validate_tbe21_packed_scan2(source)
                     or any(source.get(key, False) is not False for key in
                            ("scan", "scan4", "tabreg", "row_geometry", "wide128", "pipe128", "async_x"))):
        raise ValueError("packed_scan4 requires native schedule 1, packed_scan2 and incompatible paths off")
    return selected


def validate_tbe21_packed_scan4_environment(source, env):
    expected = validate_tbe21_packed_scan4(source)
    raw = env.get("BI_T21_PACKSCAN4", "0")
    if raw not in ("0", "1"):
        raise ValueError("TBE21 packed_scan4 environment must be exactly 0 or 1")
    if raw == "1" and (env.get("BI_TBE21_V2", "0") != "1"
                       or env.get("BI_TBE21_KERNEL_VARIANT", "2") != "1"
                       or env.get("BI_T21_PACKSCAN2", "0") != "1"
                       or any(env.get(key, "0") != "0" for key in
                              ("BI_T21_SCAN", "BI_T21_SCAN4", "BI_T21_TABREG", "BI_T21_ASYNC_X",
                               "BI_T21_ROW_GEOMETRY", "BI_T21_WIDE128", "BI_T21_PIPE128"))):
        raise ValueError("packed_scan4 environment requires native schedule 1, packed_scan2 and incompatible paths off")
    if (raw == "1") != expected:
        raise ValueError("TBE21 packed_scan4 environment differs from binding")
    return expected


def validate_tbe21_pair_vote(source):
    selected = source.get("pair_vote", False)
    if "pair_vote" in source and selected is not True:
        raise ValueError("pair_vote must be true when present")
    if selected and (validate_tbe21_kernel_variant(source.get("kernel_variant", 2)) != 1
                     or not validate_tbe21_packed_scan2(source)
                     or not validate_tbe21_tile128(source)
                     or any(source.get(key, False) is not False for key in
                            ("scan", "scan4", "packed_scan4", "row_geometry", "wide128", "pipe128", "async_x"))):
        raise ValueError("pair_vote requires schedule 1, paired scan, tile128, TABREG and incompatible paths off")
    return selected


def validate_tbe21_pair_vote_environment(source, env):
    expected = validate_tbe21_pair_vote(source)
    raw = env.get("BI_T21_PAIR_VOTE", "0")
    if raw not in ("0", "1"):
        raise ValueError("TBE21 pair_vote environment must be exactly 0 or 1")
    if raw == "1" and (env.get("BI_TBE21_V2", "0") != "1"
                       or env.get("BI_TBE21_KERNEL_VARIANT", "2") != "1"
                       or env.get("BI_T21_PACKSCAN2", "0") != "1"
                       or env.get("BI_T21_TILE128", "0") != "1"
                       or env.get("BI_T21_TABREG", "0") != "1"
                       or any(env.get(key, "0") != "0" for key in
                              ("BI_T21_SCAN", "BI_T21_SCAN4", "BI_T21_PACKSCAN4", "BI_T21_ASYNC_X",
                               "BI_T21_ROW_GEOMETRY", "BI_T21_WIDE128", "BI_T21_PIPE128"))):
        raise ValueError("pair_vote environment requires schedule 1, paired scan, tile128, TABREG and incompatible paths off")
    if (raw == "1") != expected:
        raise ValueError("TBE21 pair_vote environment differs from binding")
    return expected


def validate_tbe21_scan4(source):
    selected = source.get("scan4", False)
    if "scan4" in source and selected is not True:
        raise ValueError("scan4 must be true when present")
    if selected and not validate_tbe21_packed_scan2(source):
        raise ValueError("scan4 requires schedule 1 and packed_scan2")
    return selected


def validate_tbe21_scan4_environment(source, env):
    expected = validate_tbe21_scan4(source)
    raw = env.get("BI_T21_SCAN4", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_SCAN4 must be exactly 0 or 1")
    if raw == "1" and (env.get("BI_TBE21_V2", "0") != "1"
                       or env.get("BI_TBE21_KERNEL_VARIANT", "2") != "1"
                       or env.get("BI_T21_PACKSCAN2", "0") != "1"):
        raise ValueError("scan4 environment requires native backend, schedule 1 and packed_scan2")
    if (raw == "1") != expected:
        raise ValueError("TBE21 scan4 environment differs from binding")
    return expected


def validate_tbe21_row_geometry(source):
    selected = source.get("row_geometry", False)
    if "row_geometry" in source and selected is not True:
        raise ValueError("row_geometry must be true when present")
    if selected and not validate_tbe21_tile128(source):
        raise ValueError("row_geometry requires schedule 1, paired scan and tile128")
    return selected


def validate_tbe21_row_geometry_environment(source, env):
    expected = validate_tbe21_row_geometry(source)
    raw = env.get("BI_T21_ROW_GEOMETRY", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_ROW_GEOMETRY must be exactly 0 or 1")
    if raw == "1" and (env.get("BI_TBE21_V2", "0") != "1"
                       or env.get("BI_TBE21_KERNEL_VARIANT", "2") != "1"
                       or env.get("BI_T21_PACKSCAN2", "0") != "1"
                       or env.get("BI_T21_TILE128", "0") != "1"):
        raise ValueError("row_geometry environment requires native backend, schedule 1, paired scan and tile128")
    if (raw == "1") != expected:
        raise ValueError("TBE21 row_geometry environment differs from binding")
    return expected


def validate_tbe21_exec_layout(source):
    selected = source.get("exec_layout", False)
    if "exec_layout" in source and selected is not True:
        raise ValueError("exec_layout must be true when present")
    if selected and (validate_tbe21_kernel_variant(source.get("kernel_variant", 2)) != 1
                     or not validate_tbe21_packed_scan2(source)
                     or not validate_tbe21_tile128(source)
                     or any(source.get(key, False) is not False for key in
                            ("scan", "scan4", "packed_scan4", "async_x", "row_geometry",
                             "wide128", "pipe128", "pair_vote", "tier_prefix"))):
        raise ValueError("exec_layout requires schedule 1, paired scan, tile128 and incompatible paths off")
    return selected


def validate_tbe21_exec_layout_environment(source, env):
    expected = validate_tbe21_exec_layout(source)
    raw = env.get("BI_T21_EXEC_LAYOUT", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_EXEC_LAYOUT must be exactly 0 or 1")
    if raw == "1" and (env.get("BI_TBE21_V2", "0") != "1"
                       or env.get("BI_TBE21_KERNEL_VARIANT", "2") != "1"
                       or env.get("BI_T21_PACKSCAN2", "0") != "1"
                       or env.get("BI_T21_TILE128", "0") != "1"
                       or env.get("BI_T21_TABREG", "0") != "1"
                       or env.get("BI_T21_V1_MINBLOCKS", "2") != "2"
                       or any(env.get(key, "0") != "0" for key in
                              ("BI_T21_SCAN", "BI_T21_SCAN4", "BI_T21_PACKSCAN4", "BI_T21_ASYNC_X",
                               "BI_T21_ROW_GEOMETRY", "BI_T21_WIDE128", "BI_T21_PIPE128",
                               "BI_T21_PAIR_VOTE", "BI_T21_TIER_PREFIX", "BI_T21_TPF_PREFETCH",
                               "BI_T21_TPF_WG1", "BI_T21_TPF_PREFETCH_WG1"))):
        raise ValueError("exec_layout environment requires the exact native schedule and incompatible paths off")
    if (raw == "1") != expected:
        raise ValueError("TBE21 exec_layout environment differs from binding")
    return expected


def validate_tbe21_exec_full_g1(receipt, source):
    """Require all coded tensors through both actual EXEC producer geometries."""
    selected = validate_tbe21_exec_layout(source)
    gate = receipt.get('g1_load_gate') if isinstance(receipt, dict) else None
    record = gate.get('exec_layout') if isinstance(gate, dict) else None
    if not selected:
        if record is not None:
            raise ValueError('unbound EXEC raw-helper G1 receipt')
        return
    if not isinstance(gate, dict) or not isinstance(record, dict):
        raise ValueError('EXEC requires full selected raw-helper G1 receipt')
    count = record.get('coded_tensors')
    if (type(count) is not int or count <= 0 or gate.get('mode') != 'full'
            or type(gate.get('checked')) is not int or gate['checked'] != count
            or gate.get('mismatched') != []
            or record.get('schema') != 'codec.exec.full_g1.v1'
            or record.get('oracle') != 'selected_exec_shared_producer.v1'
            or record.get('geometries') != [64, 128]
            or any(type(value) is not int for value in record['geometries'])
            or record.get('split_policy') != 'S1+shape_selected'):
        raise ValueError('EXEC full G1 inventory or oracle differs')
    if (record.get('checked') != {'64': count, '128': count}
            or any(type(value) is not int for value in record['checked'].values())
            or record.get('mismatched') != {'64': [], '128': []}):
        raise ValueError('EXEC full G1 geometry coverage failed')
    tensors = record.get('tensors')
    if not isinstance(tensors, dict) or len(tensors) != count:
        raise ValueError('EXEC full G1 tensor coverage failed')
    for name, row in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(row, dict):
            raise ValueError('EXEC full G1 tensor record malformed')
        N, K, split = (row.get(key) for key in ('N', 'K', 'shape_split'))
        if (type(N) is not int or not 1 <= N <= 2**31-1
                or type(K) is not int or not 1 <= K <= 2**31-1 or K % 256
                or type(split) is not int or not 1 <= split <= min(8, K // 128)
                or not _encoder_sha(row.get('source_sha256'))):
            raise ValueError('EXEC full G1 shape/split/source malformed')
        splits = sorted({1, split})
        if (row.get('splits') != splits or any(type(value) is not int for value in row['splits'])
                or row.get('checked') != {'64': splits, '128': splits}
                or any(type(value) is not int for values in row['checked'].values() for value in values)
                or row.get('mismatched') != {'64': [], '128': []}):
            raise ValueError('EXEC full G1 split coverage failed')


def validate_tbe21_async_x(source):
    selected = source.get("async_x", False)
    if "async_x" in source and selected is not True:
        raise ValueError("async_x must be true when present")
    if selected and validate_tbe21_kernel_variant(source.get("kernel_variant", 2)) != 1:
        raise ValueError("async_x requires TBE21 kernel_variant 1")
    return selected

def validate_tbe21_async_x_environment(source, env):
    expected = validate_tbe21_async_x(source)
    raw = env.get("BI_T21_ASYNC_X", "0")
    if raw not in ("0", "1"):
        raise ValueError("BI_T21_ASYNC_X must be exactly 0 or 1")
    if raw == "1" and env.get("BI_TBE21_KERNEL_VARIANT", "2") != "1":
        raise ValueError("async_x environment requires TBE21 kernel_variant 1")
    if (raw == "1") != expected:
        raise ValueError("TBE21 async_x environment differs from binding")
    return expected


def validate_tbe21_kernel_variant(value):
    if type(value) is not int or value not in (1, 2):
        raise ValueError("TBE21 kernel_variant must be integer 1 or 2")
    return value


def tbe21_kernel_variant_from_environment(environ):
    value = environ.get(TBE21_KERNEL_VARIANT_ENV)
    if value is None:
        return 2
    if value not in ("1", "2"):
        raise ValueError("BI_TBE21_KERNEL_VARIANT must be unset, 1, or 2")
    return int(value)


def validate_tbe21_kernel_variant_binding(source, environ=None):
    env = os.environ if environ is None else environ
    validate_tbe21_tier_prefix_disabled(source=source, environ=env)
    validate_cpu_encoder_environment(source, env)
    validate_tbe21_tile128_environment(source, env)
    validate_tbe21_packed_scan2_environment(source, env)
    validate_tbe21_async_x_environment(source, env)
    validate_tbe21_row_geometry_environment(source, env)
    validate_tbe21_exec_layout_environment(source, env)
    validate_tbe21_scan4_environment(source, env)
    validate_tbe21_wide128_environment(source, env)
    validate_tbe21_pipe128_environment(source, env)
    validate_tbe21_packed_scan4_environment(source, env)
    validate_tbe21_pair_vote_environment(source, env)
    expected = validate_tbe21_kernel_variant(source.get("kernel_variant", 2))
    actual = tbe21_kernel_variant_from_environment(env)
    if actual != expected:
        raise ValueError("TBE21 kernel variant environment differs from qualification binding")
    return expected


def validate_overlap_build_binding(source):
    validate_tbe21_tier_prefix_disabled(source=source)
    if "overlap_tbe21_build" not in source:
        return False
    if (source.get("kind") != "context_v21_transcode"
            or source.get("backend_expectation") != "tbe21-v2"
            or source["overlap_tbe21_build"] is not True):
        raise ValueError("overlap_tbe21_build requires true and Context v21/tbe21-v2")
    return True


def validate_overlap_build_receipt(source, source_session, census=None):
    validate_tbe21_tier_prefix_disabled(source=source)
    selected = validate_overlap_build_binding(source)
    load = source_session.get("tbe21_load_receipt")
    present = isinstance(load, dict) and "overlap_tbe21_build" in load
    if not selected:
        if present:
            raise ValueError("unbound TBE21 build overlap receipt")
        return
    receipt = load.get("overlap_tbe21_build") if isinstance(load, dict) else None
    fields = {"enabled", "joined_before_ready", "plan", "backend_identity", "kernel_build",
              "elapsed_s_measured", "memory", "artifact_manifest_sha256", "source_kind",
              "serving_dependencies"}
    if (not isinstance(receipt, dict) or set(receipt) != fields
            or receipt.get("enabled") is not True or receipt.get("joined_before_ready") is not True
            or receipt.get("source_kind") != source["kind"]
            or receipt.get("artifact_manifest_sha256") != source.get("artifact_manifest_sha256")
            or source_session.get("artifact_manifest_sha256") != source.get("artifact_manifest_sha256")):
        raise ValueError("TBE21 build overlap receipt missing or source drifted")
    def digest(value):
        return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    def duration(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    plan, backend, build, memory = (receipt[key] for key in ("plan", "backend_identity", "kernel_build", "memory"))
    validate_tbe21_tier_prefix_disabled(source=source, identity=backend)
    if (not isinstance(plan, dict) or set(plan) != {"name", "build_dir", "key", "arch", "source_bindings"}
            or not digest(plan.get("key")) or not isinstance(plan.get("arch"), str) or not plan["arch"]
            or not isinstance(plan.get("name"), str) or not plan["name"].startswith("glc_bi_tbe21_v2_sm")
            or not isinstance(plan.get("build_dir"), str) or not plan["build_dir"]
            or not isinstance(plan.get("source_bindings"), dict)
            or set(plan["source_bindings"]) != TBE21_SOURCE_NAMES
            or any(not digest(value) for value in plan["source_bindings"].values())):
        raise ValueError("TBE21 build overlap plan malformed")
    if source.get("tier_prefix") is not None and plan["arch"] != "12.0":
        raise ValueError("tier-prefix build plan requires exact SM120")
    arch = plan["arch"].split(";")[0].removesuffix("+PTX").split(".")
    if (len(arch) != 2 or any(not part.isdecimal() for part in arch)
            or plan["name"] != "glc_bi_tbe21_v2_sm" + "".join(arch) + "_context" + ("_asyncx" if validate_tbe21_async_x(source) else "")
            + ("_rowgeom" if validate_tbe21_row_geometry(source) else "")
            + ("_execlayout" if validate_tbe21_exec_layout(source) else "")
            + ("_scan4" if validate_tbe21_scan4(source) else "")
            + ("_wide128" if validate_tbe21_wide128(source) else "") + _tier_profile_suffix(source)):
        raise ValueError("TBE21 build overlap target differs from module name")
    expected_packed = validate_tbe21_packed_scan2(source)
    expected_async_x = validate_tbe21_async_x(source)
    expected_variant = validate_tbe21_kernel_variant(source.get("kernel_variant", 2))
    if (not isinstance(backend, dict) or backend.get("backend") != "tbe21-v2"
            or not digest(backend.get("binary_sha256"))
            or not isinstance(backend.get("module_path"), str)
            or backend["module_path"] != str(Path(plan["build_dir"]) / (plan["name"] + ".so"))
            or backend.get("source_bindings") != plan["source_bindings"]
            or any(type(backend.get(key)) is not int or backend[key] != value
                   for key, value in (("format_code", 5), ("kqt", 0), ("spec", 0)))
            or backend.get("kernel_variant", 2) != expected_variant
            or type(backend.get("packed_scan2", False)) is not bool
            or backend.get("packed_scan2", False) != expected_packed
            or type(backend.get("async_x", False)) is not bool
            or backend.get("async_x", False) != expected_async_x
            or type(backend.get("kernel_variant", 2)) is not int):
        raise ValueError("TBE21 build overlap backend malformed")
    validate_tbe21_tile_identity(backend, validate_tbe21_tile128(source),
                                  row_geometry=validate_tbe21_row_geometry(source), exec_layout=validate_tbe21_exec_layout(source), scan4=validate_tbe21_scan4(source), wide128=validate_tbe21_wide128(source))
    if (not isinstance(build, dict) or set(build) != {"extension", "dir", "cached", "seconds", "ok"}
            or build.get("ok") is not True or type(build.get("cached")) is not bool
            or build.get("extension") != plan["name"] or build.get("dir") != plan["build_dir"]
            or not duration(build.get("seconds")) or not duration(receipt["elapsed_s_measured"])):
        raise ValueError("TBE21 build overlap lacks successful measured build")
    dependencies = receipt.get("serving_dependencies")
    targets = ("miv_gemv", "fastdec", "bidec", "bi_gemm")
    if not isinstance(dependencies, list) or len(dependencies) != len(targets):
        raise ValueError("TBE21 overlap serving dependency receipts are missing")
    for target, dep in zip(targets, dependencies):
        source_spec = dep.get("source_spec") if isinstance(dep, dict) else None
        if (not isinstance(dep, dict)
                or set(dep) != {"target", "extension", "build_dir", "cache_key", "source_spec",
                                "binary_path", "binary_sha256", "ok"}
                or dep.get("target") != target or dep.get("ok") is not True
                or not isinstance(dep.get("extension"), str) or not dep["extension"]
                or not isinstance(dep.get("build_dir"), str) or not dep["build_dir"]
                or not digest(dep.get("cache_key")) or not digest(dep.get("binary_sha256"))
                or dep.get("binary_path") != str(Path(dep["build_dir"]) / (dep["extension"] + ".so"))
                or not isinstance(source_spec, dict)
                or set(source_spec) != {"name", "sources", "headers", "inline_sources_sha256", "include_dirs",
                                        "flags", "arch", "extra"}
                or source_spec.get("name") != dep["extension"]
                or not isinstance(source_spec.get("sources"), list)
                or any(not isinstance(row, dict) or set(row) != {"path", "sha256"}
                       or not isinstance(row.get("path"), str) or not digest(row.get("sha256"))
                       for row in source_spec["sources"])
                or not isinstance(source_spec.get("headers"), list)
                or any(not isinstance(row, dict) or set(row) != {"path", "sha256"}
                       or not isinstance(row.get("path"), str) or not digest(row.get("sha256"))
                       for row in source_spec["headers"])
                or not isinstance(source_spec.get("inline_sources_sha256"), list)
                or any(not digest(value) for value in source_spec["inline_sources_sha256"])
                or not isinstance(source_spec.get("include_dirs"), list)
                or any(not isinstance(value, str) for value in source_spec["include_dirs"])
                or not isinstance(source_spec.get("flags"), list)
                or any(not isinstance(value, str) for value in source_spec["flags"])
                or not isinstance(source_spec.get("arch"), str)
                or not isinstance(source_spec.get("extra"), list)
                or any(not isinstance(row, list) or len(row) != 2
                       or any(not isinstance(value, str) for value in row) for row in source_spec["extra"])):
            raise ValueError("TBE21 overlap serving dependency receipt malformed")
    wanted_context = source.get("parallel_startup", {}).get("ram_budget_bytes", 32 << 30)
    estimates = {"context_budget_bytes_estimate": wanted_context,
        "compiler_reserve_bytes_estimate": 16 << 30, "runtime_reserve_bytes_estimate": 8 << 30,
        "required_available_bytes_estimate": wanted_context + (24 << 30)}
    if (not isinstance(memory, dict)
            or set(memory) != {*estimates, "accounting", "available_bytes_at_start"}
            or memory.get("accounting") != "estimate_not_rss"
            or any(type(memory.get(key)) is not int or memory[key] != value for key, value in estimates.items())
            or type(memory.get("available_bytes_at_start")) is not int
            or memory["available_bytes_at_start"] < estimates["required_available_bytes_estimate"]):
        raise ValueError("TBE21 overlap memory estimate drifted or lacks headroom")
    if census is not None:
        rows = census.get("descriptors", []) if isinstance(census, dict) else []
        rows = [row for row in rows if isinstance(row, dict) and row.get("format") == "tbe21"]
        if not rows or any(any(row.get(key) != backend[key] for key in
                              ("backend", "module_path", "binary_sha256", "source_bindings")) for row in rows):
            raise ValueError("TBE21 overlap compiler identity differs from actual descriptors")
        for row in rows:
            validate_tbe21_tier_prefix_disabled(source=source, identity=row)
            validate_tbe21_tile_identity(row, validate_tbe21_tile128(source),
                                         descriptor=validate_tbe21_tile128(source),
                                         row_geometry=validate_tbe21_row_geometry(source), exec_layout=validate_tbe21_exec_layout(source), scan4=validate_tbe21_scan4(source), wide128=validate_tbe21_wide128(source))
        if any(type(row.get("packed_scan2", False)) is not bool
               or row.get("packed_scan2", False) != expected_packed for row in rows):
            raise ValueError("TBE21 overlap packed_scan2 differs from actual descriptors")
        if any(type(row.get("kernel_variant", 2)) is not int
               or row.get("kernel_variant", 2) != expected_variant for row in rows):
            raise ValueError("TBE21 overlap kernel variant differs from actual descriptors")
    return receipt

PARALLEL_STARTUP_FIELDS = frozenset({"workers", "ram_budget_bytes"})
PARALLEL_STARTUP_METRIC_FIELDS = frozenset({
    "workers", "ram_budget_bytes", "max_accounted_memory_bytes_estimate",
    "prepared_tensors", "worker_pids", "worker_gpu_visible",
})
PARALLEL_STARTUP_STREAM_METRIC_FIELDS = frozenset({
    "streaming", "parent_workspace_bytes_estimate", "retained_host_bytes_estimate",
})


def validate_parallel_startup_binding(value):
    """Validate the optional protocol binding without coercing its exact fields."""
    if (not isinstance(value, dict)
            or set(value) not in (PARALLEL_STARTUP_FIELDS,
                                  PARALLEL_STARTUP_FIELDS | {"streaming"})):
        raise ValueError("parallel_startup must contain exactly workers and ram_budget_bytes, with optional streaming=true")
    if "streaming" in value and value["streaming"] is not True:
        raise ValueError("parallel_startup.streaming must be true when present")
    workers, budget = value["workers"], value["ram_budget_bytes"]
    if type(workers) is not int or not 1 <= workers <= 12:
        raise ValueError("parallel_startup.workers must be an integer from 1 through 12")
    if type(budget) is not int or budget <= 0:
        raise ValueError("parallel_startup.ram_budget_bytes must be a positive integer")
    result = {"workers": workers, "ram_budget_bytes": budget}
    if "streaming" in value:
        result["streaming"] = True
    return result


CPU_DECODER_PATHS = {
    "python_sha256": "georefine_context_runtime/bitexact_context_codec.py",
    "cpp_sha256": "georefine_context_runtime/bitexact_context_ans.cpp",
}
CPU_DECODER_ENV = ("GEOREFINE_CONTEXT_RUNTIME_COMPACT", "GEOREFINE_CONTEXT_COMPACT_RANS")


def validate_cpu_decoder_binding(source):
    if "cpu_decoder" not in source:
        return None
    value = source["cpu_decoder"]
    if source.get("kind") != "context_v21_transcode":
        raise ValueError("cpu_decoder is supported only for context_v21_transcode")
    if (not isinstance(value, dict)
            or set(value) != {"mode", *CPU_DECODER_PATHS}
            or value.get("mode") != "public_compact_rans"
            or any(type(value.get(key)) is not str or len(value[key]) != 64
                   or any(c not in "0123456789abcdef" for c in value[key])
                   for key in CPU_DECODER_PATHS)):
        raise ValueError("cpu_decoder must contain the exact public compact source identity")
    return dict(value)


def cpu_decoder_from_manifest(manifest):
    rows = manifest.get("sources")
    if not isinstance(rows, list):
        raise ValueError("cpu_decoder source manifest is malformed")
    found = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("cpu_decoder source manifest row is malformed")
        for key, path in CPU_DECODER_PATHS.items():
            if row.get("installed_path") == path:
                if key in found:
                    raise ValueError("duplicate cpu_decoder source manifest path")
                found[key] = row.get("installed_sha256")
    return validate_cpu_decoder_binding({"kind": "context_v21_transcode",
        "cpu_decoder": {"mode": "public_compact_rans", **found}})


def installed_cpu_decoder(package_root=None):
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    decoder = cpu_decoder_from_manifest(json.loads((root / MANIFEST).read_text()))
    for key, path in CPU_DECODER_PATHS.items():
        if digest_file(root / path) != decoder[key]:
            raise ValueError("cpu_decoder installed source digest mismatch")
    return decoder


def installed_cpu_encoder(package_root=None):
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    encoder = cpu_encoder_from_manifest(json.loads((root / MANIFEST).read_text()))
    for key, path in CPU_ENCODER_PATHS.items():
        if digest_file(root / path) != encoder['source_sha256'][key]:
            raise ValueError('cpu_encoder installed source digest mismatch')
    return encoder


def validate_cpu_decoder_environment(source, environ):
    expected = validate_cpu_decoder_binding(source)
    selected = environ.get(CPU_DECODER_ENV[0])
    if selected not in (None, "0", "1"):
        raise ValueError("invalid runtime compact decoder environment selector")
    if expected is None:
        if selected == "1":
            raise ValueError("runtime decoder choice is not bound by qualification protocol")
    elif any(environ.get(key) != "1" for key in CPU_DECODER_ENV):
        raise ValueError("cpu_decoder requires both compact environment selectors explicitly set to 1")
    return expected


def validate_cpu_decoder_receipt(source, load):
    validate_cpu_encoder_receipt(source, load)
    expected = validate_cpu_decoder_binding(source)
    markers = {}
    def visit(value, path=()):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "runtime_decoder":
                    markers[path + (key,)] = item
                else:
                    visit(item, path + (key,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + (index,))
    visit(load)
    if expected is None:
        if markers:
            raise ValueError("runtime decoder choice is not bound by qualification protocol")
        return
    if not isinstance(load, dict):
        raise ValueError("cpu_decoder load receipt is missing")
    wanted = {("tbe21", "runtime_decoder")}
    tbe21 = load.get("tbe21", {})
    if not isinstance(tbe21, dict):
        raise ValueError("cpu_decoder TBE21 receipt is malformed")
    transcode = tbe21.get("context_transcode", {})
    if not isinstance(transcode, dict):
        raise ValueError("cpu_decoder transcode receipt is malformed")
    if "per_tensor" in transcode:
        rows = transcode["per_tensor"]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("cpu_decoder per-tensor receipt is malformed")
        wanted.update(("tbe21", "context_transcode", "per_tensor", i, "runtime_decoder")
                      for i in range(len(rows)))
    if not wanted.issubset(markers):
        raise ValueError("cpu_decoder receipt markers are missing or unexpected")
    for record in markers.values():
        actual = validate_cpu_decoder_binding({"kind": source["kind"], "cpu_decoder": record})
        if actual != expected:
            raise ValueError("cpu_decoder runtime source identity differs from protocol")


def validate_tbe21_parallel_startup_metadata(source_session, source_binding):
    """Bind server flags to the actual loaded parallel-startup estimate receipt."""
    if not isinstance(source_session, dict) or not isinstance(source_binding, dict):
        raise ValueError("TBE21 parallel startup source metadata is missing")
    if "parallel_startup" in source_binding:
        expected = validate_parallel_startup_binding(source_binding["parallel_startup"])
    else:
        expected = None
    streaming = expected is not None and expected.get("streaming") is True
    if streaming and validate_cpu_encoder_binding(source_binding) is None:
        raise ValueError("streaming parallel startup requires a bound native_tbe21 CPU encoder")
    load_receipt = source_session.get("tbe21_load_receipt")
    validate_tbe21_kernel_variant_binding(source_binding)
    validate_cpu_decoder_environment(source_binding, os.environ)
    validate_cpu_decoder_receipt(source_binding, load_receipt)
    validate_overlap_build_receipt(source_binding, source_session)
    if expected is None:
        if isinstance(load_receipt, dict):
            tbe21 = load_receipt.get("tbe21")
            if ("context_startup" in load_receipt or
                    isinstance(tbe21, dict) and "parallel_context_startup" in tbe21):
                raise ValueError("unbound parallel Context startup appeared in runtime receipt")
        return None
    if source_binding.get("kind") != "context_v21_transcode":
        raise ValueError("parallel startup is supported only for context_v21_transcode")
    if not isinstance(load_receipt, dict):
        raise ValueError("TBE21 load receipt is missing parallel startup evidence")
    requested = load_receipt.get("context_startup")
    if (not isinstance(requested, dict)
            or set(requested) != {"requested_workers", "ram_budget_bytes"}
            or type(requested.get("requested_workers")) is not int
            or type(requested.get("ram_budget_bytes")) is not int
            or requested != {"requested_workers": expected["workers"],
                             "ram_budget_bytes": expected["ram_budget_bytes"]}):
        raise ValueError("runtime Context startup flags differ from protocol binding")
    tbe21 = load_receipt.get("tbe21")
    actual = tbe21.get("parallel_context_startup") if isinstance(tbe21, dict) else None
    wanted_metrics = (PARALLEL_STARTUP_METRIC_FIELDS | PARALLEL_STARTUP_STREAM_METRIC_FIELDS
                      if streaming else PARALLEL_STARTUP_METRIC_FIELDS)
    if not isinstance(actual, dict) or set(actual) != wanted_metrics:
        raise ValueError("runtime parallel Context startup metrics are missing or contain extra fields")
    if (type(actual.get("workers")) is not int or actual["workers"] != expected["workers"]
            or type(actual.get("ram_budget_bytes")) is not int
            or actual["ram_budget_bytes"] != expected["ram_budget_bytes"]):
        raise ValueError("runtime resolved Context startup selection differs from protocol")
    estimate = actual.get("max_accounted_memory_bytes_estimate")
    prepared = actual.get("prepared_tensors")
    pids = actual.get("worker_pids")
    if (type(estimate) is not int or not 0 < estimate <= expected["ram_budget_bytes"]
            or type(prepared) is not int or prepared < 1
            or not isinstance(pids, list) or not 1 <= len(pids) <= expected["workers"]
            or any(type(pid) is not int or pid <= 0 for pid in pids)
            or len(set(pids)) != len(pids)
            or actual.get("worker_gpu_visible") is not False):
        raise ValueError("runtime parallel Context startup metrics are invalid or drifted")
    if streaming:
        parent = actual.get("parent_workspace_bytes_estimate")
        retained = actual.get("retained_host_bytes_estimate")
        if (actual.get("streaming") is not True
                or type(parent) is not int or parent < 0
                or type(retained) is not int or retained < 0
                or parent > expected["ram_budget_bytes"]
                or retained > expected["ram_budget_bytes"]
                or parent > estimate or retained > estimate
                or parent + retained > estimate):
            raise ValueError("runtime streaming Context startup estimates are invalid or exceed accounted memory")
    return dict(actual)


def tbe21_source_binding(*,package,parent,context_typed=False,context_manifest_sha256=None,
                         context_build_root=None,backend_expectation='tbe21-v2',
                         parallel_startup=None, package_root=None, environ=None, overlap_tbe21_build=False):
    """Build the actual explicit Context-to-TBE21 protocol identity from server flags."""
    if backend_expectation!='tbe21-v2':raise ValueError('TBE21 platform qualification requires tbe21-v2')
    if type(context_typed) is not bool:raise ValueError('context_typed must be bool')
    if context_typed and context_build_root:raise ValueError('typed Context cannot have a build root')
    if context_typed and parallel_startup is not None:
        raise ValueError('parallel startup is supported only for context_v21_transcode')
    if not context_typed and not context_build_root:
        raise ValueError('platform TBE21 requires explicit Context or typed Context source')
    root=Path(parent).expanduser().resolve(strict=True)
    index=root/'model.safetensors.index.json';config=root/'config.json'
    if (index.is_symlink() or config.is_symlink() or not index.is_file() or not config.is_file()):
        raise ValueError('TBE21 parent config/index metadata missing or unsafe')
    manifest=package_manifest_path(package)
    out={'kind':'typed_context_tbe21' if context_typed else 'context_v21_transcode',
        'backend_expectation':backend_expectation,'artifact_manifest_sha256':digest_file(manifest),
        'parent':str(root),'parent_index_sha256':digest_file(index),
        'parent_config_sha256':digest_file(config)}
    if context_typed:
        if (not isinstance(context_manifest_sha256,str) or len(context_manifest_sha256)!=64
                or any(c not in '0123456789abcdef' for c in context_manifest_sha256)
                or context_manifest_sha256!=out['artifact_manifest_sha256']):
            raise ValueError('typed Context manifest must equal the opened package manifest')
        out['context_manifest_sha256']=context_manifest_sha256
    else:
        build=Path(context_build_root).expanduser().resolve()
        if ('.scratch' not in build.parts or build.is_relative_to(Path('/tmp'))
                or build.is_relative_to(Path('/private/tmp'))
                or build.is_relative_to(Path(package).expanduser().resolve())):
            raise ValueError('TBE21 Context build root must be durable .scratch outside the package')
        out['context_build_root']=str(build)
        if parallel_startup is not None:
            out['parallel_startup'] = validate_parallel_startup_binding(parallel_startup)
    env = os.environ if environ is None else environ
    from . import tbe21_tier_profile as tier
    profile = tier.from_environment(env)
    kernel_variant = tbe21_kernel_variant_from_environment(env)
    if kernel_variant != 2:
        out["kernel_variant"] = kernel_variant
    if env.get("BI_T21_PACKSCAN2", "0") == "1":
        out["packed_scan2"] = True
    validate_tbe21_packed_scan2_environment(out, env)
    if env.get("BI_T21_TILE128", "0") == "1":
        out["tile128"] = True
    validate_tbe21_tile128_environment(out, env)
    if env.get("BI_T21_ASYNC_X", "0") == "1":
        out["async_x"] = True
    validate_tbe21_async_x_environment(out, env)
    if env.get("BI_T21_ROW_GEOMETRY", "0") == "1":
        out["row_geometry"] = True
    validate_tbe21_row_geometry_environment(out, env)
    if env.get("BI_T21_EXEC_LAYOUT", "0") == "1":
        out["exec_layout"] = True
    validate_tbe21_exec_layout_environment(out, env)
    if env.get("BI_T21_SCAN4", "0") == "1":
        out["scan4"] = True
    validate_tbe21_scan4_environment(out, env)
    if env.get("BI_T21_WIDE128", "0") == "1":
        out["wide128"] = True
    validate_tbe21_wide128_environment(out, env)
    if env.get("BI_T21_PIPE128", "0") == "1":
        out["pipe128"] = True
    validate_tbe21_pipe128_environment(out, env)
    if env.get("BI_T21_PACKSCAN4", "0") == "1":
        out["packed_scan4"] = True
    validate_tbe21_packed_scan4_environment(out, env)
    if env.get("BI_T21_PAIR_VOTE", "0") == "1":
        out["pair_vote"] = True
    validate_tbe21_pair_vote_environment(out, env)
    if profile is not None:
        out["tier_prefix"] = profile
    tier.validate_environment(out, env)
    if env.get(CPU_DECODER_ENV[0]) == "1":
        if context_typed:
            raise ValueError("cpu_decoder is supported only for context_v21_transcode")
        root = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
        out["cpu_decoder"] = installed_cpu_decoder(root)
    validate_cpu_decoder_environment(out, env)
    if env.get(CPU_ENCODER_ENV) == '1':
        if context_typed:
            raise ValueError('cpu_encoder is supported only for context_v21_transcode')
        out['cpu_encoder'] = installed_cpu_encoder(package_root)
    encoder = validate_cpu_encoder_environment(out, env)
    if parallel_startup is not None:
        if out["parallel_startup"].get("streaming") is True and encoder is None:
            raise ValueError("streaming parallel startup requires a bound native_tbe21 CPU encoder")
        if encoder is not None:
            out["parallel_startup"] = {**out["parallel_startup"], "streaming": True}
    if type(overlap_tbe21_build) is not bool:
        raise ValueError("overlap_tbe21_build must be bool")
    if overlap_tbe21_build:
        out["overlap_tbe21_build"] = True
        validate_overlap_build_binding(out)
    return out


def validate_tbe21_census(census, *, source_bindings=None, kernel_variant=2, packed_scan2: bool = False, tile128: bool = False, async_x: bool = False, row_geometry: bool = False, exec_layout: bool = False, scan4: bool = False, wide128: bool = False, pipe128: bool = False, packed_scan4: bool = False, pair_vote: bool = False, tier_prefix=None):
    from glc_serve import tbe21_tier_profile as tier
    profile = tier.validate_profile(tier_prefix)
    if profile is not None:
        tier.validate_source(dict(tier_prefix=profile, kernel_variant=kernel_variant,
            packed_scan2=packed_scan2, tile128=tile128, async_x=async_x,
            row_geometry=row_geometry, exec_layout=exec_layout, scan4=scan4, wide128=wide128,
            pipe128=pipe128, packed_scan4=packed_scan4, pair_vote=pair_vote))
    if type(pair_vote) is not bool or pair_vote and (kernel_variant != 1 or not packed_scan2 or not tile128 or async_x or row_geometry or scan4 or wide128 or pipe128 or packed_scan4):
        raise ValueError("pair_vote requires schedule 1, paired scan, tile128 and incompatible paths off")
    if type(wide128) is not bool or wide128 and (kernel_variant != 1 or not packed_scan2 or not tile128 or scan4):
        raise ValueError("wide128 requires schedule 1, paired scan, tile128 and SCAN4 off")
    if type(pipe128) is not bool or pipe128 and (kernel_variant != 1 or not packed_scan2 or not tile128 or scan4 or async_x or not wide128):
        raise ValueError("pipe128 requires schedule 1, paired scan, tile128, WIDE128 on and SCAN4/ASYNC_X off")
    if type(scan4) is not bool or scan4 and (kernel_variant != 1 or not packed_scan2):
        raise ValueError("scan4 requires boolean selection, schedule 1 and packed_scan2")
    if type(packed_scan4) is not bool or packed_scan4 and (kernel_variant != 1 or not packed_scan2 or async_x or row_geometry or scan4 or wide128 or pipe128):
        raise ValueError("packed_scan4 requires schedule 1, packed_scan2 and incompatible paths off")
    if type(exec_layout) is not bool or exec_layout and (kernel_variant != 1 or not packed_scan2 or not tile128 or row_geometry or async_x or scan4 or wide128 or pipe128 or packed_scan4 or pair_vote):
        raise ValueError("exec_layout requires schedule 1, paired scan, tile128 and incompatible paths off")
    """Validate the actual TBE21 V2 dispatch census without importing Torch."""
    kernel_variant = validate_tbe21_kernel_variant(kernel_variant)
    if type(tile128) is not bool or (tile128 and (kernel_variant != 1 or not packed_scan2)):
        raise ValueError("tile128 requires kernel_variant 1 and packed_scan2")
    if type(packed_scan2) is not bool or (packed_scan2 and kernel_variant != 1):
        raise ValueError("packed_scan2 requires boolean selection and TBE21 kernel_variant 1")
    if type(async_x) is not bool or (async_x and kernel_variant != 1):
        raise ValueError("async_x requires boolean selection and TBE21 kernel_variant 1")
    if (not isinstance(census,dict) or census.get('coverage_complete') is not True
            or type(census.get('descriptor_count')) is not int
            or type(census.get('unknown_descriptor_count')) is not int
            or census['unknown_descriptor_count']!=0):
        raise ValueError('TBE21 descriptor census is incomplete')
    rows=census.get('descriptors');unknown=census.get('unknown_descriptors')
    if not isinstance(rows,list) or census['descriptor_count']!=len(rows):
        raise ValueError('TBE21 descriptor census count mismatch')
    if not isinstance(unknown,list) or unknown:
        raise ValueError('TBE21 descriptor census has unknown descriptors')
    if any(not isinstance(row,dict) for row in rows):
        raise ValueError('TBE21 descriptor census row is malformed')
    tbe=[row for row in rows if row.get('format')=='tbe21']
    if (not tbe or type(census.get('optimized_tbe21_descriptors')) is not int
            or census['optimized_tbe21_descriptors']!=len(tbe)):
        raise ValueError('TBE21 descriptor census is mixed or incomplete')
    if source_bindings is None:
        from . import tbe21_v2_backend
        source_bindings=tbe21_v2_backend.source_bindings()
    binaries=set()
    for row in tbe:
        tier.validate_identity(row, profile)
        validate_tbe21_tile_identity(row, tile128, descriptor=True, row_geometry=row_geometry, exec_layout=exec_layout, scan4=scan4, wide128=wide128, pipe128=pipe128, packed_scan4=packed_scan4)
        binary=row.get('binary_sha256');bindings=row.get('source_bindings')
        module_path=Path(row.get('module_path',''))
        if ((module_path.stem.endswith("_packscan4")) != packed_scan4
                or (module_path.stem.endswith("_pairvote")) != pair_vote
                or (module_path.stem.endswith("_execlayout")) != exec_layout):
            raise ValueError("TBE21 packed_scan4 binary identity differs from binding")
        if (row.get('backend')!='tbe21-v2'
                or row.get('kqt')!=0 or row.get('spec')!=0 or row.get('format_code')!=5
                or type(row.get("packed_scan2", False)) is not bool
                or row.get("packed_scan2", False) != packed_scan2
                or type(row.get("async_x", False)) is not bool
                or row.get("async_x", False) != async_x
                or type(row.get('kernel_variant', 2)) is not int
                or row.get('kernel_variant', 2) != kernel_variant
                or type(row.get("packed_scan4", False)) is not bool
                or row.get("packed_scan4", False) != packed_scan4
                or type(row.get("pair_vote", False)) is not bool or row.get("pair_vote", False) != pair_vote
                or type(row.get("exec_layout", False)) is not bool or row.get("exec_layout", False) != exec_layout
                or (module_path.stem.endswith("_packscan4")) != packed_scan4
                or type(row.get('dispatch_calls')) is not int or row['dispatch_calls']<=0
                or not isinstance(row.get('module_path'),str) or module_path.is_symlink()
                or not module_path.is_file()
                or not isinstance(binary,str) or len(binary)!=64 or any(c not in '0123456789abcdef' for c in binary)
                or digest_file(module_path)!=binary or bindings!=source_bindings):
            raise ValueError('TBE21 descriptor lacks the expected V2 source/binary/dispatch proof')
        binaries.add(binary)
    if len(binaries)!=1:raise ValueError('TBE21 descriptor census has mixed native binaries')
    return len(tbe)


def validate_tbe21_runtime_metadata(metadata,source):
    if isinstance(metadata, dict): validate_tbe21_tile_metadata(metadata, source)
    if not isinstance(metadata,dict) or metadata.get('weight_format')!='tbe21':
        raise ValueError('TBE21 runtime metadata format mismatch')
    loaded=metadata.get('source_session')
    if (not isinstance(loaded,dict) or loaded.get('runtime_format')!='tbe21'
            or loaded.get('source_kind')!=source.get('kind')
            or loaded.get('artifact_manifest_sha256')!=source.get('artifact_manifest_sha256')):
        raise ValueError('TBE21 runtime source or manifest differs from protocol')
    expected_config=source.get('parent_config_sha256')
    receipt=loaded.get('tbe21_load_receipt')
    asset_hashes=receipt.get('metadata_asset_sha256') if isinstance(receipt,dict) else None
    if (not isinstance(expected_config,str) or len(expected_config)!=64
            or any(c not in '0123456789abcdef' for c in expected_config)
            or not isinstance(receipt,dict)
            or receipt.get('metadata_source')!='sealed_package_assets'
            or not isinstance(asset_hashes,dict)
            or asset_hashes.get('config.json')!=expected_config):
        raise ValueError('TBE21 package config differs from bound parent config')
    if source.get('kind')=='typed_context_tbe21' and loaded.get('context_manifest_sha256')!=source.get('context_manifest_sha256'):
        raise ValueError('typed Context manifest differs from protocol')
    validate_tbe21_parallel_startup_metadata(loaded, source)
    validate_overlap_build_receipt(source, loaded, metadata.get("gemm_backend_census"))
    variant = validate_tbe21_kernel_variant(source.get("kernel_variant", 2))
    validate_tbe21_kernel_variant_binding(source)
    from . import tbe21_tier_profile as tier
    validate_tbe21_exec_full_g1(receipt, source)
    tier.validate_g1(receipt, source.get('tier_prefix'))
    return validate_tbe21_census(metadata.get('gemm_backend_census'), kernel_variant=variant, packed_scan2=source.get("packed_scan2", False), tile128=source.get("tile128", False), async_x=source.get("async_x", False), row_geometry=source.get("row_geometry", False), exec_layout=source.get("exec_layout", False), scan4=source.get("scan4", False), wide128=source.get("wide128", False), pipe128=source.get("pipe128", False), packed_scan4=source.get("packed_scan4", False), pair_vote=source.get("pair_vote", False), tier_prefix=source.get("tier_prefix"))


def _checked_metadata_file(root, name):
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError('compressed package metadata must be a regular file inside package')
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError('compressed package metadata path is unavailable') from exc
    if not resolved.is_relative_to(root):
        raise ValueError('compressed package metadata path escapes package')
    return path


def _compressed_source_parameter_count(package):
    package_path = Path(package)
    if package_path.is_symlink():
        raise ValueError('compressed package must not be a symlink')
    root = package_path.resolve()
    if not root.is_dir():
        raise ValueError('compressed package must be a directory')
    manifest_path = _checked_metadata_file(root, 'manifest.json')
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('compressed package manifest is unreadable') from exc
    from georefine_context_runtime.bitexact_predictive_package import (
        LEGACY_SCHEMA, SCHEMA, read_baseline_manifest,
    )
    if not isinstance(manifest, dict) or manifest.get('schema') not in (SCHEMA, LEGACY_SCHEMA):
        from georefine_context_runtime import typed_context_package as typed
        if isinstance(manifest, dict) and manifest.get('schema') == typed.SCHEMA:
            checked = typed.verify(root, decode_frames=False)
            return sum(math.prod(row['shape']) for row in checked['tensors'])
        raise ValueError('unsupported compressed package manifest schema')
    if manifest.get('complete') is not True:
        raise ValueError('compressed package manifest is incomplete')
    pin = manifest.get('baseline_manifest_sha256')
    if (type(pin) is not str or len(pin) != 64
            or any(c not in '0123456789abcdef' for c in pin)):
        raise ValueError('compressed package baseline manifest pin is missing or invalid')

    baseline_names = ('baseline-manifest.json', 'baseline-manifest.json.xz')
    baseline_paths = []
    for name in baseline_names:
        path = root / name
        if path.exists() or path.is_symlink():
            baseline_paths.append(_checked_metadata_file(root, name))
    if len(baseline_paths) != 1:
        raise ValueError('compressed package must contain exactly one baseline manifest')
    baseline = read_baseline_manifest(root)
    if not isinstance(baseline, dict) or baseline.get('complete') is not True:
        raise ValueError('baseline manifest is incomplete')
    if baseline.get('schema') != 'bitexact-context-package-v1':
        raise ValueError('unsupported baseline manifest schema')
    rows = baseline.get('tensors')
    if not isinstance(rows, list) or not rows:
        raise ValueError('baseline manifest has no tensor metadata')

    total = 0
    names = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('baseline tensor metadata is malformed')
        name = row.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('baseline tensor names must be unique and nonempty')
        names.add(name)
        frame = row.get('frame')
        if frame is not None:
            frame_path = PurePosixPath(frame) if isinstance(frame, str) else None
            if (frame_path is None or frame_path.is_absolute() or not frame_path.parts
                    or '..' in frame_path.parts):
                raise ValueError('baseline tensor frame path escapes package')
        shape = row.get('shape')
        if (row.get('dtype') != 'BF16' or not isinstance(shape, list) or not shape
                or any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError('invalid BF16 source shape/dtype')
        elements = math.prod(shape)
        if type(row.get('source_bytes')) is not int or row['source_bytes'] != elements * 2:
            raise ValueError('invalid source byte count')
        total += elements
    return total


def source_parameter_count(package):
    package_path = Path(package)
    manifest_path = package_path / 'manifest.json'
    if manifest_path.is_file() or manifest_path.is_symlink():
        return _compressed_source_parameter_count(package_path)
    root=package_path.resolve()
    index=json.loads(package_manifest_path(root).read_text());mapping=index['weight_map']
    headers={};payload_sizes={}
    total=0
    for name,filename in mapping.items():
        path=(root/filename).resolve()
        if not path.is_relative_to(root):raise ValueError('dense tensor path escapes package')
        if filename not in headers:
            with path.open('rb') as f:
                size=struct.unpack('<Q',f.read(8))[0]
                if not 0<size<=64*1024**2:raise ValueError('invalid safetensors header length')
                headers[filename]=json.loads(f.read(size))
                payload_sizes[filename]=path.stat().st_size-8-size
        row=headers[filename][name]
        from georefine_context_runtime.typed_context_package import _source_descriptor
        try:
            _source_descriptor(row['dtype'], row['shape'], row['data_offsets'], payload_sizes[filename], name)
        except ValueError as exc:
            raise ValueError('dense source shape does not match physical payload') from exc
        total+=math.prod(row['shape'])
    return total


def _safe_path(name):
    if not isinstance(name,str):raise ValueError('invalid installed source path')
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] not in PACKAGES:
        raise ValueError('source path escapes installed packages')
    return path


def verify_installed_wheel(wheel,package_root):
    root=Path(package_root).resolve();manifest_bytes=(root/MANIFEST).read_bytes()
    data=json.loads(manifest_bytes);sources={}
    for row in data['sources']:
        name=row['installed_path'];_safe_path(name)
        if name in sources:raise ValueError('duplicate installed source path')
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('installed source missing or outside root')
        expected=row['installed_sha256']
        if digest_file(path)!=expected:raise ValueError('installed source digest mismatch: '+name)
        sources[name]=expected
    if not sources:raise ValueError('empty source manifest')
    actual={p.relative_to(root).as_posix() for package in PACKAGES for p in (root/package).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if actual!=set(sources)|{MANIFEST}:raise ValueError('installed package closure differs from manifest')
    with zipfile.ZipFile(wheel) as archive:
        names=archive.namelist()
        if len(set(names))!=len(names):raise ValueError('duplicate wheel member')
        if archive.read(MANIFEST)!=manifest_bytes:raise ValueError('wheel and installed source manifests differ')
        shipped={n for n in names if not n.endswith('/') and PurePosixPath(n).parts[0] in PACKAGES}
        if shipped!=set(sources)|{MANIFEST}:raise ValueError('wheel source closure differs from manifest')
        for name,expected in sources.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=expected:raise ValueError('wheel source digest mismatch: '+name)
    return hashlib.sha256(manifest_bytes).hexdigest()


def resident_startup_selection(protocol, *, role, backend, metal_operator):
    """Carry opt-in metadata from the already parsed and hash-bound protocol."""
    if 'startup_disposition' not in protocol:
        return None
    selection = protocol['startup_disposition']
    if (protocol.get('acceptance_target') != 'a100_40'
            or protocol.get('target_mode') != 'candidate_capacity'
            or protocol.get('qualification_scope') != 'full_model'
            or role != 'candidate' or backend != 'cuda' or metal_operator != 'tbe21'
            or not isinstance(selection, dict)
            or set(selection) != {'policy_id', 'policy_sha256', 'owner_record_sha256'}
            or selection.get('policy_id') != 'a100_resident_capacity_20261028'
            or any(not isinstance(selection.get(k), str) or len(selection[k]) != 64
                   or any(c not in '0123456789abcdef' for c in selection[k])
                   for k in ('policy_sha256', 'owner_record_sha256'))):
        raise ValueError('resident startup receipt requires bound A100 candidate-capacity selection')
    return dict(selection)


def bind_protocol(protocol_path,*,role,package,wheel=None,package_root=None,exact_digest=False,
                  max_requests=1,device='cpu',metal_operator='decode',backend='mlx',spec_k=0,
                  fused_batching=False,max_batch_rows=128,panels=None,tune=None,tbe21_source=None,
                  mtp_confirm_kv_only=False, adaptive_serving=None):
    validate_cpu_encoder_environment(tbe21_source if isinstance(tbe21_source, dict) else {}, os.environ)
    if role not in ('candidate','reference'):raise ValueError('invalid qualification role')
    raw=Path(protocol_path).read_bytes();protocol=json.loads(raw)
    if protocol.get('schema')!='georefine.platform_qualify.protocol.v1':raise ValueError('invalid qualification protocol schema')
    if (backend not in ('mlx','cuda') or type(spec_k) is not int or not 0<=spec_k<=7 or
        type(fused_batching) is not bool or type(max_batch_rows) is not int or max_batch_rows<1):
        raise ValueError('invalid qualification runtime identity')
    phase='exactness' if exact_digest else 'performance'
    resident_selection = resident_startup_selection(
        protocol, role=role, backend=backend, metal_operator=metal_operator)
    expected=protocol['engines'][role][phase]['expected_identity']
    if (type(mtp_confirm_kv_only) is not bool
            or (mtp_confirm_kv_only and (backend!='cuda' or spec_k<=0))
            or expected.get('mtp_confirm_kv_only',False) is not mtp_confirm_kv_only):
        raise ValueError('qualification identity mismatch: mtp_confirm_kv_only')
    wheel_path=Path(wheel or protocol['artifacts']['candidate_wheel']['path'])
    root=Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    source_hash=verify_installed_wheel(wheel_path,root)
    adaptive = expected.get('adaptive_serving')
    if adaptive is not None or adaptive_serving is not None:
        if not isinstance(adaptive, dict) or backend != 'cuda' or adaptive_serving != adaptive.get('objective'):
            raise ValueError('adaptive qualification requires its frozen resource contract')
        trusted = adaptive_worklist_sources_from_manifest(json.loads((root / MANIFEST).read_text()))
        validate_adaptive_worklist_expectation(adaptive, trusted)
        if (adaptive['controls']['max_rows'] != max_batch_rows
                or adaptive['controls']['spec_k'] != spec_k
                or adaptive['controls']['max_slots'] != max_requests):
            raise ValueError('adaptive qualification declared runtime differs from frozen controls')
    validate_cpu_decoder_environment(tbe21_source if isinstance(tbe21_source, dict) else {}, os.environ)
    pool=protocol['prompt_pool'];pool_hash=digest_file(pool['path'])
    if pool_hash!=pool['sha256']:raise ValueError('prompt pool digest mismatch')
    frame=protocol['frame_id'];tokens=protocol['max_tokens'];concurrency=protocol['concurrency']
    if not isinstance(frame,str) or not frame or type(tokens) is not int or tokens<1:
        raise ValueError('invalid qualification frame')
    if not isinstance(concurrency,list) or not concurrency or any(type(n) is not int or n<1 for n in concurrency):
        raise ValueError('invalid qualification concurrency')
    if max_requests<max(concurrency):raise ValueError('server request capacity is below qualification concurrency')
    identity=dict(candidate_wheel_sha256=digest_file(wheel_path),source_manifest_sha256=source_hash,
                  package_manifest_sha256=digest_file(package_manifest_path(package)),frame_id=frame,
                  prompt_pool_sha256=pool_hash,max_tokens=tokens,concurrency=concurrency,
                  backend=backend,spec_k=spec_k,fused_batching=fused_batching,
                  max_batch_rows=max_batch_rows,
                  panel_inventory_sha256=digest_file(panels) if panels is not None else None)
    if adaptive is not None:
        identity['adaptive_serving'] = adaptive
    if mtp_confirm_kv_only:
        identity['mtp_confirm_kv_only']=True
    if metal_operator=='tbe21':
        if backend!='cuda' or not isinstance(tbe21_source,dict):
            raise ValueError('TBE21 requires an explicit CUDA source/backend identity')
        encoder = validate_cpu_encoder_environment(tbe21_source, os.environ)
        if encoder is not None and encoder != installed_cpu_encoder(root):
            raise ValueError('cpu_encoder differs from installed/wheel source manifest')
        decoder = validate_cpu_decoder_environment(tbe21_source, os.environ)
        if decoder is not None and decoder != cpu_decoder_from_manifest(json.loads((root / MANIFEST).read_text())):
            raise ValueError('cpu_decoder differs from installed/wheel source manifest')
        if expected.get('tbe21_source')!=tbe21_source:
            raise ValueError('qualification identity mismatch: tbe21_source')
        if tbe21_source.get('backend_expectation')!='tbe21-v2':
            raise ValueError('TBE21 requires explicit tbe21-v2 backend expectation')
        identity['tbe21_source']=dict(tbe21_source)
    elif tbe21_source is not None or 'tbe21_source' in expected:
        raise ValueError('tbe21_source requires metal_operator=tbe21')
    if backend=='cuda':
        if tune is None:raise ValueError('CUDA qualification requires an explicit pinned tune')
        identity['tune_sha256']=digest_file(tune)
    if 'source_parameter_count' in expected:
        identity['source_parameter_count']=source_parameter_count(package)
    for name,value in identity.items():
        if name in ('backend','spec_k','fused_batching','max_batch_rows','panel_inventory_sha256','tune_sha256'):
            if name not in expected or type(expected[name]) is not type(value) or expected[name]!=value:
                raise ValueError('qualification identity mismatch: '+name)
        elif expected.get(name)!=value:raise ValueError('qualification identity mismatch: '+name)
    identity.update(protocol_sha256=hashlib.sha256(raw).hexdigest(),qualification_role=role,
                    pid=os.getpid(),host=platform.node(),system=platform.system(),machine=platform.machine(),
                    exact_digest=bool(exact_digest),device=device,metal_operator=metal_operator,
                    qualification_scope='artifact identity and declared workload; measured results are separate')
    for name in ('device','metal_operator'):
        if name in expected and expected[name]!=identity[name]:
            raise ValueError('qualification identity mismatch: '+name)
    if resident_selection is not None:
        identity['resident_startup_selection'] = resident_selection
    return identity


ADAPTIVE_WORKLIST_SOURCES = frozenset({
    'predictive_batch.py', 'serving_resources.py', 'state_budget.py',
    'bidec_policy.py', 'bidec.py'})
ADAPTIVE_WORKLIST_CONTROLS = ('max_slots', 'max_rows', 'pages_total', 'max_ctx',
                              'spec_k', 'prefill_priority', 'item_cap')


def adaptive_worklist_sources_from_manifest(manifest):
    """Extract trusted hashes from an already verified installed/source manifest."""
    found = {}
    for row in manifest.get('sources', []):
        name = row.get('installed_path') if isinstance(row, dict) else None
        if isinstance(name, str) and name.startswith('glc_serve/'):
            short = name.removeprefix('glc_serve/')
            if short in ADAPTIVE_WORKLIST_SOURCES:
                value = row.get('installed_sha256')
                if short in found or not _encoder_sha(value):
                    raise ValueError('adaptive work-list source manifest invalid or duplicated')
                found[short] = value
    if set(found) != ADAPTIVE_WORKLIST_SOURCES:
        raise ValueError('adaptive work-list source manifest incomplete')
    return found


def _adaptive_integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('adaptive work-list invalid integer: ' + name)
    return value


def _adaptive_same(actual, expected, label):
    # JSON type equality distinguishes 1 from true, including nested records.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('adaptive work-list differs: ' + label)


def validate_adaptive_worklist_expectation(value, trusted_sources):
    """Validate and replay a frozen, source-bound admission contract; no torch/CUDA."""
    from dataclasses import fields
    from . import state_budget as sb
    from .serving_resources import native_worklist_item_cap
    if (not isinstance(value, dict) or set(value) != {'schema', 'objective', 'controls',
            'ledger', 'reserve_bytes', 'source_bindings'}
            or value['schema'] != 'georefine.adaptive_worklist_expectation.v1'
            or value['objective'] not in ('capacity', 'prefill')):
        raise ValueError('adaptive work-list frozen expectation missing or malformed')
    if (not isinstance(trusted_sources, dict) or set(trusted_sources) != ADAPTIVE_WORKLIST_SOURCES
            or any(not _encoder_sha(v) for v in trusted_sources.values())):
        raise ValueError('adaptive work-list trusted sources missing')
    _adaptive_same(value['source_bindings'], trusted_sources, 'trusted planner sources')
    controls = value['controls']
    if not isinstance(controls, dict) or set(controls) != set(ADAPTIVE_WORKLIST_CONTROLS):
        raise ValueError('adaptive work-list controls missing')
    for name in ADAPTIVE_WORKLIST_CONTROLS:
        if name != 'prefill_priority':
            _adaptive_integer(controls[name], name, 0 if name == 'spec_k' else 1)
    priority = controls['prefill_priority']
    import math
    if (type(priority) not in (int, float) or not math.isfinite(priority) or not 0 <= priority <= 1
            or (value['objective'] == 'prefill' and priority <= 0)):
        raise ValueError('adaptive work-list prefill priority invalid')
    if (controls['max_rows'] not in sb.BUCKETS or controls['spec_k'] > 7
            or controls['max_slots'] * (controls['spec_k'] + 1) > controls['max_rows']
            or controls['pages_total'] != controls['max_slots'] * ((controls['max_ctx'] + sb.PAGE - 1) // sb.PAGE + 1)
            or controls['item_cap'] != native_worklist_item_cap(
                pages_total=controls['pages_total'], max_rows=controls['max_rows'],
                max_ctx=controls['max_ctx'], prefill_priority=priority)):
        raise ValueError('adaptive work-list frozen geometry/capacity invalid')
    record = value['ledger']
    if not isinstance(record, dict) or set(record) != {'shapes', 'config', 'extra_workspace_bytes',
            'mm_rope_rows', 'mm_device_bytes', 'ledger_device_bytes'}:
        raise ValueError('adaptive work-list replay ledger missing')
    shapes, config = record['shapes'], record['config']
    if (not isinstance(shapes, dict) or set(shapes) != {f.name for f in fields(sb.Shapes)}
            or not isinstance(config, dict) or set(config) != {f.name for f in fields(sb.EngineConfig)}):
        raise ValueError('adaptive work-list ledger fields invalid')
    for name, number in shapes.items():
        if name == 'name':
            if not isinstance(number, str) or not number:
                raise ValueError('adaptive work-list model name invalid')
        elif name == 'gated':
            if type(number) is not bool:
                raise ValueError('adaptive work-list gated shape invalid')
        else:
            _adaptive_integer(number, 'shape.' + name)
    for name in ('H', 'V', 'F', 'n_att', 'NQ', 'NKV', 'D', 'rope_dim'):
        _adaptive_integer(shapes[name], 'shape.' + name, 1)
    mapping = {'slots':'max_slots', 'pages':'pages_total', 'max_rows':'max_rows',
               'max_ctx':'max_ctx', 'spec_k':'spec_k', 'item_cap':'item_cap'}
    for key, control in mapping.items():
        _adaptive_same(config[key], controls[control], 'ledger.' + key)
    for key in ('dummy_state', 'fd_buffers_live', 'mtp_enabled'):
        if type(config[key]) is not bool:
            raise ValueError('adaptive work-list ledger bool invalid')
    _adaptive_integer(config['prefix_cache_mb'], 'prefix_cache_mb')
    if config['dummy_state'] or config['fd_buffers_live'] or config['prefix_cache_mb'] != 0:
        raise ValueError('adaptive work-list ledger must price the native adaptive configuration')
    for key in ('conv_ring', 'sms', 'fd_max_len'):
        _adaptive_integer(config[key], 'config.' + key, 1)
    if config['conv_ring'] != 16:
        raise ValueError('adaptive work-list native conv ring mismatch')
    for key in ('extra_workspace_bytes', 'mm_rope_rows', 'mm_device_bytes', 'ledger_device_bytes'):
        _adaptive_integer(record[key], key)
    expected_mm = record['mm_rope_rows'] * shapes['rope_dim'] * 4
    if record['mm_rope_rows']:
        expected_mm += controls['max_rows'] * shapes['H'] * 2
    if record['mm_device_bytes'] != expected_mm:
        raise ValueError('adaptive work-list multimodal ledger charge differs')
    native = sb.summarize(sb.ledger(sb.Shapes(**shapes), sb.EngineConfig(**config)))['total_bytes']
    if record['ledger_device_bytes'] != native:
        raise ValueError('adaptive work-list native ledger charge differs')
    _adaptive_integer(value['reserve_bytes'], 'reserve_bytes')
    return native + record['extra_workspace_bytes'] + expected_mm


def validate_adaptive_worklist_plan(plan, controls, expected, trusted_sources):
    """Fail before allocation if the chosen profile differs from the frozen admission."""
    total = validate_adaptive_worklist_expectation(expected, trusted_sources)
    if not isinstance(plan, dict) or plan.get('mode') != 'adaptive':
        raise ValueError('adaptive work-list admitted plan missing')
    _adaptive_same(plan.get('source_bindings'), trusted_sources, 'plan sources')
    _adaptive_same(plan.get('objective'), expected['objective'], 'objective')
    selected, effective = plan.get('selected', {}), plan.get('effective_controls', {})
    if not all(isinstance(record, dict) for record in (selected, effective, controls)):
        raise ValueError('adaptive work-list controls/selected profile missing')
    for name, value in expected['controls'].items():
        for label, record in [('runtime', controls), ('selected', selected), ('effective', effective)]:
            _adaptive_same(record.get(name), value, label + '.' + name)
    _adaptive_same(selected.get('ledger'), expected['ledger'], 'admitted ledger inputs')
    _adaptive_same(selected.get('estimated_engine_bytes'), total, 'estimated engine bytes')
    free = _adaptive_integer(plan.get('free_bytes'), 'free_bytes')
    device_total = _adaptive_integer(plan.get('device_total_bytes'), 'device_total_bytes', 1)
    if free > device_total:
        raise ValueError('adaptive work-list invalid physical memory')
    for name in ('available_bytes', 'credited_available_bytes'):
        _adaptive_same(plan.get(name), free, name)
    for name in ('reclaimable_bytes', 'admission_cache_credit_bytes'):
        _adaptive_same(plan.get(name), 0, name)
    _adaptive_integer(plan.get('observed_allocator_unused_reserved_bytes'), 'allocator slack')
    reserve = expected['reserve_bytes']
    _adaptive_same(plan.get('reserve_bytes'), reserve, 'reserve_bytes')
    _adaptive_same(plan.get('engine_budget_bytes'), max(0, free - reserve), 'engine budget')
    remaining = free - reserve - total
    if remaining < 0:
        raise ValueError('adaptive work-list allocation spends the physical reserve')
    _adaptive_same(selected.get('remaining_after_engine_and_reserve_bytes'), remaining, 'selected remaining')
    _adaptive_same(plan.get('conservative_remaining_after_engine_and_reserve_bytes'), remaining, 'remaining')
    _adaptive_same(plan.get('sms'), expected['ledger']['config']['sms'], 'device SM count')
    _adaptive_same(plan.get('pages_total'), expected['controls']['pages_total'], 'plan pages')
    return total


def adaptive_worklist_allocation(bd):
    """Read actual native dimensions and array sizes; never infer bytes from the cap alone."""
    from .state_budget import PAGE
    cap, nq, dim = bd.g_cap, bd.NQ, bd.head_dim
    for name, value in [('item_cap', cap), ('NQ', nq), ('D', dim)]:
        _adaptive_integer(value, name, 1)
    def array_bytes(tensor, shape, dtype):
        if tuple(tensor.shape) != shape or str(tensor.dtype) != dtype:
            raise ValueError('adaptive work-list native allocation shape/dtype differs')
        return tensor.numel() * tensor.element_size()
    pacc = array_bytes(bd.pacc, (cap, nq, dim), 'torch.float32')
    pml = array_bytes(bd.pml, (cap, nq, 2), 'torch.float32')
    device = sum(array_bytes(getattr(bd.meta, key), (cap,), 'torch.int32')
                 for key in ('item_row', 'item_split'))
    host = 0
    for key in ('item_row', 'item_split'):
        array = bd.meta.h(key)
        if tuple(array.shape) != (cap,) or str(array.dtype) != 'int32':
            raise ValueError('adaptive work-list host allocation shape/dtype differs')
        host += array.nbytes
    if bd.state_bytes.get('pacc') != pacc + pml:
        raise ValueError('adaptive work-list native state byte telemetry differs')
    return dict(item_cap=cap, NQ=nq, D=dim, PAGE=PAGE, pacc_bytes=pacc, pml_bytes=pml,
                item_metadata_device_bytes=device, item_metadata_pinned_host_bytes=host)


def validate_adaptive_worklist_metadata(runtime, expected=None, trusted_sources=None, *, actual=None):
    """Validate using trusted protocol selection, not a removable receipt selector."""
    if expected is None:
        if isinstance(runtime, dict) and isinstance(runtime.get('page_plan'), dict) and runtime['page_plan'].get('mode') == 'adaptive':
            raise ValueError('adaptive work-list runtime is unbound')
        return None
    if not isinstance(runtime, dict):
        raise ValueError('adaptive work-list runtime missing')
    total = validate_adaptive_worklist_plan(runtime.get('page_plan'), runtime, expected, trusted_sources)
    cap = expected['controls']['item_cap']; shapes = expected['ledger']['shapes']
    nq, dim = shapes['NQ'], shapes['D']
    from .state_budget import PAGE
    wanted = dict(item_cap=cap, NQ=nq, D=dim, PAGE=PAGE,
                  pacc_bytes=cap*nq*dim*4, pml_bytes=cap*nq*2*4,
                  item_metadata_device_bytes=cap*8, item_metadata_pinned_host_bytes=cap*8)
    _adaptive_same(runtime.get('worklist_allocation'), wanted, 'allocation dimensions/bytes')
    _adaptive_same(runtime.get('state_bytes', {}).get('pacc'), wanted['pacc_bytes']+wanted['pml_bytes'], 'state pacc bytes')
    if actual is not None:
        _adaptive_same(adaptive_worklist_allocation(actual), wanted, 'actual native arrays')
        for name in ('max_slots', 'max_rows', 'pages_total', 'max_ctx'):
            _adaptive_same(getattr(actual, name), expected['controls'][name], 'actual.' + name)
        _adaptive_same(actual.NQ, shapes['NQ'], 'actual loaded NQ')
        _adaptive_same(actual.head_dim, shapes['D'], 'actual loaded D')
    return total


def validate_adaptive_worklist_launch(expected, *, objective, max_users, max_rows,
                                      max_ctx, spec_k, prefill_priority, reserve_bytes):
    """Match explicit launch choices to the frozen contract, without allocating."""
    if expected is None:
        if objective is not None:
            raise ValueError('adaptive qualification requires a frozen resource contract')
        return
    validate_adaptive_worklist_expectation(expected, expected.get('source_bindings') if isinstance(expected, dict) else None)
    _adaptive_same(objective, expected['objective'], 'launch objective')
    controls = dict(max_slots=max_users, max_rows=max_rows, max_ctx=max_ctx,
                    spec_k=spec_k, prefill_priority=prefill_priority)
    for name, value in controls.items():
        # CLI floating point spelling may differ, but bool is never numeric input.
        target = expected['controls'][name]
        if type(value) not in (int, float) or value != target:
            raise ValueError('adaptive work-list launch differs: ' + name)
    _adaptive_same(reserve_bytes, expected['reserve_bytes'], 'launch reserve')


def validate_adaptive_worklist_argv(argv, expected):
    """Require explicit dimensions for the adaptive formal launch; legacy is untouched."""
    selected = [v for v in argv if isinstance(v, str) and
                (v == '--adaptive-serving' or v.startswith('--adaptive-serving='))]
    if expected is None and not selected:
        return
    if expected is None:
        raise ValueError('adaptive launch has no trusted protocol contract')
    def argument(flag, converter, default=None):
        values = []
        for index, arg in enumerate(argv):
            if arg == flag:
                if index + 1 == len(argv):
                    raise ValueError('adaptive launch flag has no value: ' + flag)
                values.append(argv[index + 1])
            elif isinstance(arg, str) and arg.startswith(flag + '='):
                values.append(arg.split('=', 1)[1])
        if not values and default is not None:
            return default
        if len(values) != 1:
            raise ValueError('adaptive launch requires one explicit ' + flag)
        try:
            return converter(values[0])
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('adaptive launch invalid ' + flag) from exc
    validate_adaptive_worklist_launch(expected,
        objective=argument('--adaptive-serving', str),
        max_users=argument('--max-users', int), max_rows=argument('--max-rows', int),
        max_ctx=argument('--max-context', int), spec_k=argument('--spec-k', int),
        prefill_priority=argument('--prefill-priority', float),
        reserve_bytes=argument('--vram-headroom-gib', lambda v: int(round(float(v) * (1 << 30)))))
    if argument('--pages-total', int, 0) != 0:
        raise ValueError('adaptive launch must retain automatic page admission')
