"""Нижняя граница времени по вычислениям и чтению (главы 1.3.2, 2.2.2, 4.8).

При полном перекрытии чтения и вычислений время шага не меньше
max(F/Π, R/β). Это нижняя граница: реальные ядра, синхронизация и очередь
её не достигают, и разрыв объясняется измерением (раздел 1.3.4).
"""

from __future__ import annotations

import math


def _positive(name: str, value: float) -> None:
    if not value > 0:
        raise ValueError(f"{name} должен быть больше нуля: {value}")


def _non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} не может быть отрицательным: {value}")


def compute_seconds(flops: float, peak: float) -> float:
    _non_negative("flops", flops)
    _positive("peak", peak)
    return flops / peak


def memory_seconds(bytes_moved: float, bandwidth: float) -> float:
    _non_negative("bytes_moved", bytes_moved)
    _positive("bandwidth", bandwidth)
    return bytes_moved / bandwidth


def lower_bound_seconds(
    flops: float, bytes_moved: float, peak: float, bandwidth: float
) -> float:
    return max(compute_seconds(flops, peak), memory_seconds(bytes_moved, bandwidth))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    _non_negative("flops", flops)
    _positive("bytes_moved", bytes_moved)
    return flops / bytes_moved


def ridge_point(peak: float, bandwidth: float) -> float:
    _positive("peak", peak)
    _positive("bandwidth", bandwidth)
    return peak / bandwidth


def batch_threshold(
    shared_read_bytes: float, kv_bytes_per_token: float, context_tokens: int
) -> int:
    """Минимальный батч, при котором чтение контекста догоняет чтение общих весов (пример 2-2)."""
    _non_negative("shared_read_bytes", shared_read_bytes)
    _positive("kv_bytes_per_token", kv_bytes_per_token)
    _positive("context_tokens", context_tokens)
    return math.ceil(shared_read_bytes / (kv_bytes_per_token * context_tokens))
