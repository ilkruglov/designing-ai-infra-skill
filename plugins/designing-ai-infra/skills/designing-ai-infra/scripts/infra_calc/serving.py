"""Память, нижние границы TTFT и TPOT, пропускная способность и цена токена (главы 3, 8)."""

from __future__ import annotations

import math

from .checks import require_int_at_least, require_non_negative, require_positive
from .roofline import lower_bound_seconds


def max_concurrent_requests(
    memory_bytes: float,
    weight_bytes: float,
    kv_bytes_per_request: float,
    reserve_bytes: float = 0,
) -> int:
    """Сколько запросов помещается по памяти; 0, если веса и резерв не помещаются."""
    require_non_negative("memory_bytes", memory_bytes)
    require_non_negative("weight_bytes", weight_bytes)
    require_non_negative("reserve_bytes", reserve_bytes)
    require_positive("kv_bytes_per_request", kv_bytes_per_request)
    free = memory_bytes - weight_bytes - reserve_bytes
    if free <= 0:
        return 0
    return math.floor(free / kv_bytes_per_request)


def ttft_lower_bound_seconds(
    prefill_flops: float, weight_read_bytes: float, peak: float, bandwidth: float
) -> float:
    return lower_bound_seconds(prefill_flops, weight_read_bytes, peak, bandwidth)


def tpot_lower_bound_seconds(
    batch: int,
    decode_flops_per_request: float,
    weight_read_bytes: float,
    kv_read_bytes_per_request: float,
    peak: float,
    bandwidth: float,
) -> float:
    require_int_at_least("batch", batch, 1)
    require_non_negative("decode_flops_per_request", decode_flops_per_request)
    require_non_negative("weight_read_bytes", weight_read_bytes)
    require_non_negative("kv_read_bytes_per_request", kv_read_bytes_per_request)
    return lower_bound_seconds(
        batch * decode_flops_per_request,
        weight_read_bytes + batch * kv_read_bytes_per_request,
        peak,
        bandwidth,
    )


def tokens_per_second(batch: int, step_seconds: float) -> float:
    require_int_at_least("batch", batch, 0)
    require_positive("step_seconds", step_seconds)
    return batch / step_seconds


def cost_per_million_tokens(price_per_hour: float, tokens_per_second: float) -> float:
    require_non_negative("price_per_hour", price_per_hour)
    require_positive("tokens_per_second", tokens_per_second)
    return price_per_hour / 3600 / tokens_per_second * 1e6
