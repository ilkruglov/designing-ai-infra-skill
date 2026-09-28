"""Коллективные коммуникации (глава 6.4).

Кольцевой AllReduce: T = 2(n−1)α + 2(n−1)M/(nB), формула (6-9).
При малых сообщениях время определяют запуски раундов, а не пропускная способность.
"""

from __future__ import annotations

import math

from .roofline import _non_negative, _positive


def _devices(name: str, devices: int) -> None:
    if devices < 1:
        raise ValueError(f"{name} должно быть не меньше 1: {devices}")


def ring_bytes_sent_per_device(devices: int, message_bytes: float) -> float:
    _devices("devices", devices)
    _non_negative("message_bytes", message_bytes)
    return 2 * (devices - 1) * message_bytes / devices


def ring_allreduce_seconds(
    devices: int, message_bytes: float, bandwidth: float, round_latency: float
) -> float:
    _positive("bandwidth", bandwidth)
    _non_negative("round_latency", round_latency)
    return (
        2 * (devices - 1) * round_latency
        + ring_bytes_sent_per_device(devices, message_bytes) / bandwidth
    )


def tree_allreduce_rounds(devices: int) -> int:
    _devices("devices", devices)
    return 2 * math.ceil(math.log2(devices))


def tp_step_seconds(
    local_seconds_single: float,
    tp: int,
    reductions: int,
    message_bytes: float,
    bandwidth: float,
    round_latency: float,
) -> float:
    """Шаг при TP: локальная часть делится на tp, добавляются кольцевые редукции (формулы 6-5, 6-9)."""
    _non_negative("local_seconds_single", local_seconds_single)
    _devices("tp", tp)
    _non_negative("reductions", reductions)
    _non_negative("message_bytes", message_bytes)
    _positive("bandwidth", bandwidth)
    _non_negative("round_latency", round_latency)
    communication = (
        0.0
        if tp == 1
        else reductions
        * ring_allreduce_seconds(tp, message_bytes, bandwidth, round_latency)
    )
    return local_seconds_single / tp + communication


def all_to_all_phase(
    counts: list[list[int]], vector_bytes: int, bandwidth: float, startup_seconds: float
) -> tuple[int, float, float]:
    """Попарный all-to-all по матрице назначений (calculations/src/infra_calc/topics/all_to_all.py).

    counts[i][j] — сколько векторов ранг i отправляет рангу j; локальные не идут в сеть.
    Возвращает байты в сети, нижнюю границу по самой загруженной конечной точке
    и время попарной схемы с барьером после каждого раунда.
    """
    p = len(counts)
    if p == 0:
        raise ValueError("матрица назначений пуста")
    for i, row in enumerate(counts):
        if len(row) != p:
            raise ValueError(
                f"матрица назначений не квадратная: в строке {i} {len(row)} элементов вместо {p}"
            )
        for j, count in enumerate(row):
            _non_negative(f"counts[{i}][{j}]", count)
    _non_negative("vector_bytes", vector_bytes)
    _positive("bandwidth", bandwidth)
    _non_negative("startup_seconds", startup_seconds)
    sends = [
        sum(row[j] for j in range(p) if j != i) * vector_bytes
        for i, row in enumerate(counts)
    ]
    receives = [
        sum(counts[i][j] for i in range(p) if i != j) * vector_bytes for j in range(p)
    ]
    pairwise = 0.0
    for offset in range(1, p):
        sizes = [
            counts[i][(i + offset) % p] * vector_bytes
            for i in range(p)
            if counts[i][(i + offset) % p]
        ]
        if sizes:
            pairwise += startup_seconds + max(sizes) / bandwidth
    return sum(sends), max(sends + receives) / bandwidth, pairwise
