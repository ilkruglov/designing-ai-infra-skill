"""Обучение: конвейер, checkpoint, состояние и срок (главы 3.4, 10.3, 10.4).

Пузырь конвейера (p − 1)(t_f + t_b) и утилизация m/(m + p − 1) — по книге
(глава 10.3.2) и calculations/src/infra_calc/topics/training_pipeline_schedule.py
оригинала (github.com/bojieli/ai-infra-book, пин 3bdcb4fc). Чередующийся 1F1B
с v виртуальными стадиями делит пузырь на v: доля (1/v)(p − 1)/m.

Интервал checkpoint: модель первого порядка c/τ + λτ/2 + λr с оптимумом
sqrt(2c/λ) и точный оптимум пуассоновской модели восстановления из
calculations/src/infra_calc/topics/checkpoint_interval.py оригинала. Частота
сбоев задания λ = N/MTBF + 1/MTBF_job: общий для задания поток событий (откат
из-за всплеска потерь, глава 10.4.5; common_job_mtbf у автора) прибавляется один
раз, а не на каждое устройство.
"""

from __future__ import annotations

import math

from .checks import require_int_at_least, require_non_negative, require_positive


def _check_pipeline(stages: int, microbatches: int, virtual_stages: int) -> None:
    require_int_at_least("stages", stages, 1)
    require_int_at_least("microbatches", microbatches, 1)
    require_int_at_least("virtual_stages", virtual_stages, 1)
    if virtual_stages > 1 and microbatches % stages:
        # порядок Megatron-LM, которому следует автор: training_pipeline_schedule.py
        raise ValueError(
            f"чередующийся 1F1B требует число micro-batch, кратное числу стадий: "
            f"microbatches={microbatches}, stages={stages}"
        )


def pipeline_utilization(
    stages: int, microbatches: int, virtual_stages: int = 1
) -> float:
    """u = m/(m + (p − 1)/v); при v = 1 — m/(m + p − 1) для fill–drain и 1F1B (глава 10.3.2).

    Предполагаются одинаковые стадии и нулевые передачи между ними и обновление:
    в примере книги 330 ms без передач становятся 337 ms с ними, а 1F1B — 347 ms.
    """
    _check_pipeline(stages, microbatches, virtual_stages)
    return microbatches / (microbatches + (stages - 1) / virtual_stages)


def pipeline_bubble_ratio(
    stages: int, microbatches: int, virtual_stages: int = 1
) -> float:
    """Доля пузыря относительно полезной работы: (1/v)(p − 1)/m (глава 10.3.2)."""
    _check_pipeline(stages, microbatches, virtual_stages)
    return (stages - 1) / (virtual_stages * microbatches)


def pipeline_bubble_seconds(
    stages: int,
    forward_seconds: float,
    backward_seconds: float,
    virtual_stages: int = 1,
) -> float:
    """Простой каждой стадии (p − 1)(t_f + t_b)/v без передач и обновления."""
    require_int_at_least("stages", stages, 1)
    require_int_at_least("virtual_stages", virtual_stages, 1)
    require_non_negative("forward_seconds", forward_seconds)
    require_non_negative("backward_seconds", backward_seconds)
    return (stages - 1) * (forward_seconds + backward_seconds) / virtual_stages


def pipeline_step_seconds(
    stages: int,
    microbatches: int,
    forward_seconds: float,
    backward_seconds: float,
    virtual_stages: int = 1,
) -> float:
    """(m + (p − 1)/v)(t_f + t_b): нижняя граница шага при сбалансированных стадиях.

    Передачи через границы стадий и обновление параметров не учитываются: в
    примере главы 10.3.2 формула даёт 330 ms, событийная модель — 337 ms для
    fill–drain, 347 ms для 1F1B и 298 ms для чередующегося 1F1B при v = 2.
    """
    _check_pipeline(stages, microbatches, virtual_stages)
    require_non_negative("forward_seconds", forward_seconds)
    require_non_negative("backward_seconds", backward_seconds)
    return (microbatches + (stages - 1) / virtual_stages) * (
        forward_seconds + backward_seconds
    )


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


def job_failure_rate(
    devices: int, device_mtbf: float, common_job_mtbf: float | None = None
) -> float:
    """λ = N/MTBF_device + 1/MTBF_job (главы 10.4.4 и 10.4.5).

    Сбой любого устройства прерывает задание, поэтому устройства складываются;
    общий для задания поток (например, откат из-за всплеска потерь) прибавляется
    один раз. Оба потока предполагаются независимыми и пуассоновскими.
    """
    require_int_at_least("devices", devices, 1)
    require_positive("device_mtbf", device_mtbf)
    rate = devices / device_mtbf
    if common_job_mtbf is not None:
        require_positive("common_job_mtbf", common_job_mtbf)
        rate += 1 / common_job_mtbf
    return rate


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


def sharded_state_components(
    parameters: int,
    devices: int,
    stage: int,
    weight_bytes: float = 2,
    grad_bytes: float = 2,
    optimizer_bytes: float = 12,
) -> tuple[int, int, int]:
    """Веса, градиенты и состояние оптимизатора на одном GPU при ZeRO-0..3 (глава 10.2.1).

    ZeRO-1 шардирует состояние оптимизатора (основные веса FP32 и два момента
    Adam), ZeRO-2 — ещё и градиенты, ZeRO-3 — ещё и веса. Шард — 1/devices;
    активации, буферы AllGather и временные полные градиенты не входят.
    """
    require_int_at_least("parameters", parameters, 0)
    require_int_at_least("devices", devices, 1)
    require_int_at_least("stage", stage, 0)
    require_non_negative("weight_bytes", weight_bytes)
    require_non_negative("grad_bytes", grad_bytes)
    require_non_negative("optimizer_bytes", optimizer_bytes)
    if stage > 3:
        raise ValueError(f"stage ZeRO должен быть 0, 1, 2 или 3: {stage}")

    def share(bytes_per_param: float, sharded: bool) -> int:
        per_param = bytes_per_param / devices if sharded else bytes_per_param
        return int(parameters * per_param)

    return (
        share(weight_bytes, stage >= 3),
        share(grad_bytes, stage >= 2),
        share(optimizer_bytes, stage >= 1),
    )


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
    return sum(
        sharded_state_components(
            parameters, devices, stage, weight_bytes, grad_bytes, optimizer_bytes
        )
    )


def training_seconds(
    total_flops: float, devices: int, peak: float, mfu: float
) -> float:
    require_non_negative("total_flops", total_flops)
    require_int_at_least("devices", devices, 1)
    require_positive("peak", peak)
    if not 0 < mfu <= 1:
        raise ValueError(f"MFU должен лежать в (0, 1]: {mfu}")
    return total_flops / (devices * peak * mfu)
