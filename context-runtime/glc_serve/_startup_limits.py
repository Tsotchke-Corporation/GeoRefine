"""Resource limits shared by opt-in verified CPU startup entry points."""

MAX_CPU_ENCODE_WORKERS = 16


def cpu_resolver_cache_limit(workers: int) -> int:
    """Retain a bounded reference working set as explicit concurrency grows."""
    if type(workers) is not int or not 1 <= workers <= MAX_CPU_ENCODE_WORKERS:
        raise ValueError("invalid CPU worker count")
    return 512 * 1024**2 * ((workers + 3) // 4)
