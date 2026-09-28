"""Нижняя граница времени по вычислениям и чтению (главы 1.3.2, 2.2.2, 4.8).

При полном перекрытии чтения и вычислений время шага не меньше
max(F/Π, R/β). Это нижняя граница: реальные ядра, синхронизация и очередь
её не достигают, и разрыв объясняется измерением (раздел 1.3.4).
"""

from __future__ import annotations

import math

from .checks import require_int_at_least, require_non_negative, require_positive


def compute_seconds(flops: float, peak: float) -> float:
    require_non_negative("flops", flops)
    require_positive("peak", peak)
    return flops / peak


def memory_seconds(bytes_moved: float, bandwidth: float) -> float:
    require_non_negative("bytes_moved", bytes_moved)
    require_positive("bandwidth", bandwidth)
    return bytes_moved / bandwidth


def lower_bound_seconds(
    flops: float, bytes_moved: float, peak: float, bandwidth: float
) -> float:
    return max(compute_seconds(flops, peak), memory_seconds(bytes_moved, bandwidth))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    require_non_negative("flops", flops)
    require_positive("bytes_moved", bytes_moved)
    return flops / bytes_moved


def ridge_point(peak: float, bandwidth: float) -> float:
    require_positive("peak", peak)
    require_positive("bandwidth", bandwidth)
    return peak / bandwidth


def batch_threshold(
    shared_read_bytes: float, kv_bytes_per_token: float, context_tokens: int
) -> int:
    """Минимальный батч, при котором чтение контекста догоняет чтение общих весов (пример 2-2)."""
    require_non_negative("shared_read_bytes", shared_read_bytes)
    require_positive("kv_bytes_per_token", kv_bytes_per_token)
    require_int_at_least("context_tokens", context_tokens, 1)
    return math.ceil(shared_read_bytes / (kv_bytes_per_token * context_tokens))


def compute_bound_batch(
    decode_flops: float,
    weight_read_bytes: float,
    kv_read_bytes_per_request: float,
    peak: float,
    bandwidth: float,
) -> float | None:
    """Батч B_*, с которого B·F/Π ≥ (R_W + B·R_KV)/β (глава 1.3.2).

    B_* = (R_W/β) / (F/Π − R_KV/β); без KV это b_W·Π/(2β) книги. None, если
    на запрос F/Π ≤ R_KV/β: чтение контекста растёт с батчем не медленнее
    вычислений, и шаг decode не становится вычислительно ограниченным.
    Ёмкость памяти не проверяется.
    """
    require_non_negative("decode_flops", decode_flops)
    require_non_negative("weight_read_bytes", weight_read_bytes)
    require_non_negative("kv_read_bytes_per_request", kv_read_bytes_per_request)
    require_positive("peak", peak)
    require_positive("bandwidth", bandwidth)
    gap = decode_flops / peak - kv_read_bytes_per_request / bandwidth
    if gap <= 0:
        return None
    return weight_read_bytes / bandwidth / gap
