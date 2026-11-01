"""Device-name-free, opt-in admission over the native engine's exact state ledger.

The caller supplies memory left after resident weights, actual SM count to its
ledger callback, and an explicit reserve for unmodelled graph/allocator costs.
This chooses feasible capacity, not an unmeasured throughput winner.
"""
from __future__ import annotations

from typing import Callable

from .state_budget import BUCKETS, PAGE


class ResourceAdmissionError(ValueError):
    """No supported full-context configuration fits the declared memory budget."""

    def __init__(self, detail):
        self.detail = detail
        super().__init__(
            f"adaptive serving cannot admit {detail['min_users']} full-context users: "
            f"{detail['available_bytes']} bytes available, "
            f"{detail['reserve_bytes']} reserved, minimum engine "
            f"{detail['minimum_engine_bytes']} bytes; context and precision were not reduced")


def _integer(name, value, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def native_worklist_item_cap(*, pages_total: int, max_rows: int, max_ctx: int,
                             prefill_priority: float = 0.0) -> int:
    """Adaptive row/context envelope, including speculative verify and padded rows.

    Each valid real row costs at most ceil(max_ctx/PAGE) items; padding costs
    one. A captured bucket never exceeds max_rows. The native ledger prices
    this exact cap, shared by trunk and MTP, before resource admission.
    """
    for name, value in [('pages_total', pages_total), ('max_rows', max_rows),
                        ('max_ctx', max_ctx)]:
        _integer(name, value, positive=True)
    from .bidec_policy import plan_item_cap

    policy_cap = plan_item_cap(pages_total, max_rows, max_ctx,
                              prefill_priority=prefill_priority)['item_cap']
    return max(policy_cap, max_rows * ((max_ctx + PAGE - 1) // PAGE))


def plan_native_resources(*, free_bytes: int, total_bytes: int, reclaimable_bytes: int,
                          reserve_bytes: int, max_users: int, min_users: int,
                          max_rows: int, max_ctx: int, spec_k: int,
                          estimate_bytes: Callable[[int, int, int], int],
                          objective: str = 'capacity') -> dict:
    """Admit full-context slots and rows from bounded native geometries.

    ``estimate_bytes(users, pages, rows)`` must return the engine state ledger
    plus its model-specific workspace and multimodal staging costs. It must
    be nondecreasing in users for a fixed row geometry. Explicit reserves are
    never spent on those buffers. One trunk-plus-draft row per user must fit
    the selected capture geometry. No dtype, context or split policy changes.
    """
    for name, value in [('free_bytes', free_bytes), ('reclaimable_bytes', reclaimable_bytes),
                        ('reserve_bytes', reserve_bytes), ('spec_k', spec_k)]:
        _integer(name, value)
    for name, value in [('total_bytes', total_bytes), ('max_users', max_users),
                        ('min_users', min_users), ('max_rows', max_rows), ('max_ctx', max_ctx)]:
        _integer(name, value, positive=True)
    if free_bytes + reclaimable_bytes > total_bytes:
        raise ValueError('free plus reclaimable memory exceeds device total')
    if min_users > max_users:
        raise ValueError('min_users exceeds requested max_users')
    if objective not in ('capacity', 'prefill'):
        raise ValueError('adaptive objective must be capacity or prefill')
    rows_allowed = [r for r in BUCKETS if r <= max_rows and r >= min_users * (spec_k + 1)]
    if not rows_allowed:
        raise ValueError('requested row ceiling cannot hold the minimum full speculative workload')
    available = free_bytes + reclaimable_bytes
    budget = max(0, available - reserve_bytes)
    per_user_pages = (max_ctx + PAGE - 1) // PAGE + 1
    evaluated = 0

    def profile(users, rows):
        nonlocal evaluated
        pages = users * per_user_pages
        size = estimate_bytes(users, pages, rows)
        _integer('estimated engine bytes', size)
        evaluated += 1
        return {'max_slots': users, 'max_rows': rows, 'pages_total': pages,
                'max_ctx': max_ctx, 'spec_k': spec_k,
                'estimated_engine_bytes': size,
                'remaining_after_engine_and_reserve_bytes': available - reserve_bytes - size}

    minimum = profile(min_users, rows_allowed[0])
    profiles = []
    for rows in rows_allowed:
        upper = min(max_users, rows // (spec_k + 1))
        low, high = min_users, upper
        best = None
        while low <= high:
            users = (low + high) // 2
            candidate = profile(users, rows)
            if available >= reserve_bytes and candidate['estimated_engine_bytes'] <= budget:
                best = candidate
                low = users + 1
            else:
                high = users - 1
        if best is not None:
            profiles.append(best)
    if not profiles:
        raise ResourceAdmissionError({
            'min_users': min_users, 'max_ctx': max_ctx, 'spec_k': spec_k,
            'available_bytes': available, 'reserve_bytes': reserve_bytes,
            'minimum_engine_bytes': minimum['estimated_engine_bytes'],
            'minimum_rows': minimum['max_rows'],
        })
    frontier = [p for p in profiles if not any(
        other['max_slots'] >= p['max_slots'] and other['max_rows'] >= p['max_rows']
        and (other['max_slots'] > p['max_slots'] or other['max_rows'] > p['max_rows'])
        for other in profiles)]
    keys = ('max_slots', 'max_rows') if objective == 'capacity' else ('max_rows', 'max_slots')
    selected = max(frontier, key=lambda row: tuple(row[k] for k in keys))
    return {'schema': 'georefine.adaptive_serving_resources.v1', 'objective': objective,
            'selected': selected, 'frontier': frontier, 'profiles': profiles,
            'free_bytes': free_bytes, 'reclaimable_bytes': reclaimable_bytes,
            'device_total_bytes': total_bytes, 'available_bytes': available,
            'reserve_bytes': reserve_bytes, 'engine_budget_bytes': budget,
            'requested': {'max_users': max_users, 'min_users': min_users,
                          'max_rows': max_rows, 'max_ctx': max_ctx, 'spec_k': spec_k},
            'pages_per_full_context_user': per_user_pages, 'evaluations': evaluated,
            'performance_status': 'UNMEASURED',
            'unmodelled': 'Graph pools, CUDA context, allocator slack and other costs must fit reserve_bytes.'}
