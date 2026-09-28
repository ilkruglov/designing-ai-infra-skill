"""Обучение: конвейер, checkpoint, состояние и срок (главы 3.4, 10.3, 10.4).

Интервал checkpoint: модель первого порядка c/τ + λτ/2 + λr с оптимумом
sqrt(2c/λ) и точный оптимум пуассоновской модели восстановления из
calculations/src/infra_calc/topics/checkpoint_interval.py оригинала.
"""

from __future__ import annotations

import math

from .roofline import _non_negative, _positive


def _pipeline(stages: int, microbatches: int) -> None:
    if stages < 1:
        raise ValueError(f"стадий конвейера должно быть не меньше 1: {stages}")
    if microbatches < 1:
        raise ValueError(f"микропакетов должно быть не меньше 1: {microbatches}")


def _devices(devices: int) -> None:
    if devices < 1:
        raise ValueError(f"устройств должно быть не меньше 1: {devices}")


def pipeline_utilization(stages: int, microbatches: int) -> float:
    _pipeline(stages, microbatches)
    return microbatches / (microbatches + stages - 1)


def fill_drain_seconds(
    stages: int, microbatches: int, forward_seconds: float, backward_seconds: float
) -> float:
    _pipeline(stages, microbatches)
    _non_negative("forward_seconds", forward_seconds)
    _non_negative("backward_seconds", backward_seconds)
    return (microbatches + stages - 1) * (forward_seconds + backward_seconds)


def checkpoint_first_order_loss(
    interval: float, save_seconds: float, failure_rate: float, recovery_seconds: float
) -> float:
    _positive("interval", interval)
    _non_negative("save_seconds", save_seconds)
    _non_negative("failure_rate", failure_rate)
    _non_negative("recovery_seconds", recovery_seconds)
    return (
        save_seconds / interval
        + failure_rate * interval / 2
        + failure_rate * recovery_seconds
    )


def checkpoint_optimal_interval(save_seconds: float, failure_rate: float) -> float:
    _positive("save_seconds", save_seconds)
    _positive("failure_rate", failure_rate)
    return math.sqrt(2 * save_seconds / failure_rate)


def checkpoint_poisson_optimal_interval(
    save_seconds: float, failure_rate: float
) -> float:
    """Единственный положительный корень y − 1 + exp(−y − λc) = 0, τ = y/λ."""
    _positive("save_seconds", save_seconds)
    _positive("failure_rate", failure_rate)
    a = failure_rate * save_seconds
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if middle + math.expm1(-middle - a) < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2 / failure_rate


def state_bytes(parameters: int, bytes_per_param: float) -> int:
    _non_negative("parameters", parameters)
    _non_negative("bytes_per_param", bytes_per_param)
    return int(parameters * bytes_per_param)


def sharded_state_bytes_per_device(
    parameters: int,
    devices: int,
    stage: int,
    weight_bytes: float = 2,
    grad_bytes: float = 2,
    optimizer_bytes: float = 12,
) -> int:
    """Постоянно размещённое состояние обучения на одном GPU при ZeRO-0..3 (глава 10.2.1).

    optimizer_bytes по умолчанию — основные веса FP32 и два момента Adam FP32.
    """
    _non_negative("parameters", parameters)
    _devices(devices)
    _non_negative("weight_bytes", weight_bytes)
    _non_negative("grad_bytes", grad_bytes)
    _non_negative("optimizer_bytes", optimizer_bytes)
    w, g, o = weight_bytes, grad_bytes, optimizer_bytes
    per_param = {
        0: w + g + o,
        1: w + g + o / devices,
        2: w + (g + o) / devices,
        3: (w + g + o) / devices,
    }
    if stage not in per_param:
        raise ValueError(f"stage ZeRO должен быть 0, 1, 2 или 3: {stage}")
    return int(parameters * per_param[stage])


def training_seconds(
    total_flops: float, devices: int, peak: float, mfu: float
) -> float:
    _non_negative("total_flops", total_flops)
    _devices(devices)
    _positive("peak", peak)
    if not 0 < mfu <= 1:
        raise ValueError(f"MFU должен лежать в (0, 1]: {mfu}")
    return total_flops / (devices * peak * mfu)
