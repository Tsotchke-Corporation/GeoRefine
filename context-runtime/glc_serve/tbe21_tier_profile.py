"""Pure identity contract for explicitly selected experimental TBE21 tier profiles.

A matching identity authorizes comparison, never a correctness/performance verdict.
"""
from __future__ import annotations

from pathlib import Path

ENV_FIELDS = ('BI_T21_TIER_PREFIX', 'BI_T21_TPF_PREFETCH',
              'BI_T21_TPF_WG1', 'BI_T21_TPF_PREFETCH_WG1')
ROW_GEOMETRY_ENV = 'BI_T21_ROW_GEOMETRY'
IDENTITY_FIELDS = ('tier_prefix', 'tier_prefetch', 'tier_prefix_wg1', 'tier_prefetch_wg1')
PROFILE_FIELDS = frozenset(('abi', 'prefetch', 'wg1', 'prefetch_wg1'))
METADATA_FIELDS = ('tier_metadata_abi', 'tier_decode_default_wg', 'tier_decode_geometries')
INCOMPATIBLE = ('scan', 'scan4', 'packed_scan4', 'async_x', 'wide128', 'pipe128', 'pair_vote')
CLI_PROFILES = {
    'tier-prefix': dict(abi=1, prefetch=False, wg1=False, prefetch_wg1=False),
    'tier-prefix-wg1': dict(abi=1, prefetch=False, wg1=True, prefetch_wg1=False),
    'tier-prefetch': dict(abi=1, prefetch=True, wg1=False, prefetch_wg1=False),
    'tier-prefix-wg1-prefetch-wg2': dict(abi=1, prefetch=True, wg1=True, prefetch_wg1=False),
    'tier-prefetch-wg1': dict(abi=1, prefetch=True, wg1=True, prefetch_wg1=True),
}


def validate_profile(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != PROFILE_FIELDS
            or type(value.get('abi')) is not int or value['abi'] != 1
            or any(type(value.get(k)) is not bool for k in ('prefetch', 'wg1', 'prefetch_wg1'))
            or value['prefetch_wg1'] and not (value['prefetch'] and value['wg1'])):
        raise ValueError('tier-prefix profile requires exact ABI1 and boolean prefetch/WG1 controls')
    return dict(value)


def validate_source(source):
    if 'tier_prefix' not in source:
        if 'row_geometry' in source and type(source['row_geometry']) is not bool:
            raise ValueError('row-geometry source binding must be boolean')
        return None
    if source['tier_prefix'] is None:
        raise ValueError('tier-prefix source binding must omit the disabled profile')
    value = validate_profile(source['tier_prefix'])
    if 'row_geometry' in source and type(source['row_geometry']) is not bool:
        raise ValueError('tier-prefix row_geometry source binding must be boolean')
    if (type(source.get('kernel_variant')) is not int or source['kernel_variant'] != 1
            or source.get('packed_scan2') is not True or source.get('tile128') is not True
            or any(source.get(k, False) is not False for k in INCOMPATIBLE)):
        raise ValueError('tier-prefix source requires schedule1, paired scan, tile128 and incompatible experiments off')
    return value


def from_environment(env):
    if env.get(ROW_GEOMETRY_ENV, '0') not in ('0', '1'):
        raise ValueError(ROW_GEOMETRY_ENV + ' must be exactly 0 or 1')
    for name in ENV_FIELDS:
        if env.get(name, '0') not in ('0', '1'):
            raise ValueError(name + ' must be exactly 0 or 1')
    enabled = env.get(ENV_FIELDS[0], '0') == '1'
    if not enabled:
        if any(env.get(name, '0') == '1' for name in ENV_FIELDS[1:]):
            raise ValueError('tier-prefix submodes require BI_T21_TIER_PREFIX=1')
        return None
    required = {'BI_TBE21_V2': '1', 'BI_TBE21_KERNEL_VARIANT': '1',
                'BI_T21_PACKSCAN2': '1', 'BI_T21_TILE128': '1', 'BI_T21_TABREG': '1'}
    incompatible = ('BI_T21_SCAN', 'BI_T21_SCAN4', 'BI_T21_PACKSCAN4', 'BI_T21_ASYNC_X',
                    'BI_T21_WIDE128', 'BI_T21_PIPE128', 'BI_T21_PAIR_VOTE')
    if (any(env.get(name, '0') != value for name, value in required.items())
            or env.get('BI_T21_V1_MINBLOCKS', '2') != '2'
            or any(env.get(name, '0') != '0' for name in incompatible)):
        raise ValueError('tier-prefix environment requires the paired register-table schedule and incompatible experiments off')
    return validate_profile(dict(abi=1, prefetch=env.get(ENV_FIELDS[1], '0') == '1',
                                 wg1=env.get(ENV_FIELDS[2], '0') == '1',
                                 prefetch_wg1=env.get(ENV_FIELDS[3], '0') == '1'))


def validate_environment(source, env):
    expected = validate_source(source)
    actual = from_environment(env)
    if actual != expected:
        raise ValueError('tier-prefix environment differs from explicit source binding')
    source_row_geometry = source.get('row_geometry', False)
    if type(source_row_geometry) is not bool or (env.get(ROW_GEOMETRY_ENV, '0') == '1') != source_row_geometry:
        raise ValueError('row_geometry environment differs from exact source binding')
    if expected is not None and env.get('BIDEC_TBE2_G1') != 'full':
        raise ValueError('tier-prefix qualification requires BIDEC_TBE2_G1=full')
    return expected


def suffix(profile):
    profile = validate_profile(profile)
    if profile is None:
        return ''
    return '_tpf' + ''.join(char if profile[key] else '0'
                           for char, key in (('p', 'prefetch'), ('w', 'wg1'), ('q', 'prefetch_wg1')))


def identity_profile(identity):
    if not isinstance(identity, dict) or not isinstance(identity.get('module_path', ''), str):
        raise ValueError('tier-prefix descriptor identity must have a string module path')
    enabled = identity.get('tier_prefix', False)
    if type(enabled) is not bool:
        raise ValueError('tier-prefix descriptor selection must be boolean')
    if not enabled:
        row_geometry = identity.get('row_geometry', False)
        if type(row_geometry) is not bool:
            raise ValueError('row-geometry descriptor selection must be boolean')
        if (any(name in identity and (type(identity[name]) is not bool or identity[name])
                for name in IDENTITY_FIELDS)
                or any(name in identity for name in METADATA_FIELDS)
                or '_tpf' in Path(identity.get('module_path', '')).stem):
            raise ValueError('tier-prefix descriptor identity is not bound')
        return None
    values = {}
    module_stem = Path(identity.get('module_path', '')).stem
    if 'row_geometry' not in identity and '_rowgeom' in module_stem:
        raise ValueError('combined tier-prefix descriptor lacks explicit row_geometry')
    row_geometry = identity.get('row_geometry', False)
    if type(row_geometry) is not bool:
        raise ValueError('tier-prefix descriptor row_geometry must be boolean')
    for name, key in zip(IDENTITY_FIELDS[1:], ('prefetch', 'wg1', 'prefetch_wg1')):
        if type(identity.get(name)) is not bool:
            raise ValueError('tier-prefix descriptor lacks exact boolean controls')
        values[key] = identity[name]
    profile = validate_profile(dict(abi=identity.get('tier_metadata_abi'), **values))
    geometries = [1, 2] if profile['wg1'] else [2]
    actual = identity.get('tier_decode_geometries')
    if (type(identity.get('tier_decode_default_wg')) is not int
            or identity['tier_decode_default_wg'] != 2
            or not isinstance(actual, list) or any(type(x) is not int for x in actual)
            or actual != geometries
            or Path(identity.get('module_path', '')).stem != ('glc_bi_tbe21_v2_sm120_context'
                + ('_rowgeom' if row_geometry else '') + suffix(profile))):
        raise ValueError('tier-prefix descriptor geometry or exact SM120 binary differs from binding')
    return profile


def validate_identity(identity, profile):
    expected = validate_profile(profile)
    if identity_profile(identity) != expected:
        raise ValueError('tier-prefix descriptor differs from explicit profile binding')
    return expected


def geometries(profile):
    profile = validate_profile(profile)
    return ([1, 2] if profile['wg1'] else [2]) if profile is not None else []


def validate_g1(load_receipt, profile):
    expected = validate_profile(profile)
    gate = load_receipt.get('g1_load_gate') if isinstance(load_receipt, dict) else None
    if expected is None:
        if isinstance(gate, dict) and 'tier_prefix' in gate:
            raise ValueError('tier-prefix G1 evidence lacks an explicit profile binding')
        return
    actual = gate.get('tier_prefix') if isinstance(gate, dict) else None
    keys = {'profile', 'coded_tensors', 'geometries', 'checked', 'mismatched'}
    if (not isinstance(actual, dict) or set(actual) != keys or gate.get('mode') != 'full'
            or gate.get('mismatched') != [] or type(gate.get('checked')) is not int
            or type(actual.get('coded_tensors')) is not int or actual['coded_tensors'] <= 0
            or gate['checked'] != actual['coded_tensors']
            or validate_profile(actual.get('profile')) != expected
            or actual.get('geometries') != geometries(expected)
            or any(type(v) is not int for v in actual.get('geometries', []))):
        raise ValueError('tier-prefix requires full source-hash G1 for every enabled geometry')
    names = {str(x) for x in geometries(expected)}
    checked, mismatched = actual.get('checked'), actual.get('mismatched')
    if (not isinstance(checked, dict) or set(checked) != names
            or any(type(v) is not int or v != actual['coded_tensors'] for v in checked.values())
            or not isinstance(mismatched, dict) or set(mismatched) != names
            or any(type(v) is not list or v for v in mismatched.values())):
        raise ValueError('tier-prefix G1 geometry coverage is incomplete or mismatched')
