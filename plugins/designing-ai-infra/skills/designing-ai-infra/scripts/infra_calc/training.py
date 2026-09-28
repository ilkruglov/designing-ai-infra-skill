"""Обучение: конвейер, checkpoint, состояние и срок (главы 3.4, 10.3, 10.4).

Пузырь конвейера (p − 1)(t_f + t_b) и утилизация m/(m + p − 1) — по книге
(глава 10.3.2) и calculations/src/infra_calc/topics/training_pipeline_schedule.py
оригинала (github.com/bojieli/ai-infra-book, пин 56ecb425).

Интервал checkpoint: модель первого порядка c/τ + λτ/2 + λr с оптимумом
sqrt(2c/λ) и точный оптимум пуассоновской модели восстановления из
calculations/src/infra_calc/topics/checkpoint_interval.py оригинала.
"""

from __future__ import annotations

import math

from .checks import require_int_at_least, require_non_negative, require_positive


def pipeline_utilization(stages: int, microbatches: int) -> float:
    """u = m/(m + p − 1) для fill–drain и 1F1B (глава 10.3.2).

    Предполагаются одинаковые стадии и нулевые передачи между ними и обновление:
    в примере книги 330 ms без передач становятся 337 ms с ними, а 1F1B — 347 ms.
    """
    require_int_at_least("stages", stages, 1)
    require_int_at_least("microbatches", microbatches, 1)
    return microbatches / (microbatches + stages - 1)


def fill_drain_seconds(
    stages: int, microbatches: int, forward_seconds: float, backward_seconds: float
) -> float:
    """(m + p − 1)(t_f + t_b): шаг fill–drain при сбалансированных стадиях.

    Передачи через границы стадий и обновление параметров не учитываются
    (глава 10.3.2: 330 ms против 337 ms с ними; 1F1B при тех же условиях — 347 ms).
    """
    require_int_at_least("stages", stages, 1)
    require_int_at_least("microbatches", microbatches, 1)
    require_non_negative("forward_seconds", forward_seconds)
    require_non_negative("backward_seconds", backward_seconds)
    return (microbatches + stages - 1) * (forward_seconds + backward_seconds)


def checkpoint_first_order_loss(
    interval: float, save_seconds: float, failure_rate: float, recovery_seconds: float
) -> float:
    """Доля потерь первого порядка c/τ + λτ/2 + λr.

    τ — полезное время между сохранениями без учёта самой паузы c; результат
    больше 1 не обрезается и не является вероятностью (как у автора).
    """
    require_positive("interval", interval)
    require_non_negative("save_seconds", save_seconds)
    require_non_negative("failure_rate", failure_rate)
    require_non_negative("recovery_seconds", recovery_seconds)
    return (
        save_seconds / interval
        + failure_rate * interval / 2
        + failure_rate * recovery_seconds
    )


def checkpoint_optimal_interval(save_seconds: float, failure_rate: float) -> float:
    require_positive("save_seconds", save_seconds)
    require_positive("failure_rate", failure_rate)
    return math.sqrt(2 * save_seconds / failure_rate)


def checkpoint_poisson_optimal_interval(
    save_seconds: float, failure_rate: float
) -> float:
    """Единственный положительный корень y − 1 + exp(−y − λc) = 0, τ = y/λ."""
    require_positive("save_seconds", save_seconds)
    require_positive("failure_rate", failure_rate)
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
    require_int_at_least("parameters", parameters, 0)
    require_non_negative("bytes_per_param", bytes_per_param)
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

    По умолчанию 16 байт на параметр, как в главе 10: веса BF16 (2), градиенты
    BF16 (2), основные веса FP32 и два момента Adam FP32 (12). Глава 3.4.1 считает
    градиенты в FP32 и получает 18 байт; для неё передайте grad_bytes=4.
    """
    require_int_at_least("parameters", parameters, 0)
    require_int_at_least("devices", devices, 1)
    require_int_at_least("stage", stage, 0)
    require_non_negative("weight_bytes", weight_bytes)
    require_non_negative("grad_bytes", grad_bytes)
    require_non_negative("optimizer_bytes", optimizer_bytes)
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
    require_non_negative("total_flops", total_flops)
    require_int_at_least("devices", devices, 1)
    require_positive("peak", peak)
    if not 0 < mfu <= 1:
        raise ValueError(f"MFU должен лежать в (0, 1]: {mfu}")
    return total_flops / (devices * peak * mfu)
