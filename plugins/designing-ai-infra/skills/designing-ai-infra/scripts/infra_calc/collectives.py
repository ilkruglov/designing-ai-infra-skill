"""Коллективные коммуникации (глава 6.4).

Кольцевой AllReduce: T = 2(n−1)α + 2(n−1)M/(nB), формула (6-9).
При малых сообщениях время определяют запуски раундов, а не пропускная способность.

Формулы кольца и дерева перенесены из calculations/src/infra_calc/topics/
ring_collective.py и tree_collective.py оригинала (github.com/bojieli/ai-infra-book,
пин d0cc188b), all-to-all — из all_to_all.py там же.
"""

from __future__ import annotations

import math

from .checks import require_int_at_least, require_non_negative, require_positive


def ring_bytes_sent_per_device(devices: int, message_bytes: float) -> float:
    require_int_at_least("devices", devices, 1)
    require_non_negative("message_bytes", message_bytes)
    return 2 * (devices - 1) * message_bytes / devices


def ring_allreduce_seconds(
    devices: int, message_bytes: float, bandwidth: float, round_latency: float
) -> float:
    require_positive("bandwidth", bandwidth)
    require_non_negative("round_latency", round_latency)
    return (
        2 * (devices - 1) * round_latency
        + ring_bytes_sent_per_device(devices, message_bytes) / bandwidth
    )


def tree_allreduce_rounds(devices: int) -> int:
    """2·ceil(log2 n) раундов несегментированного биномиального дерева.

    Формула (6-10) книги записана для n — степени двойки; ceil — расширение на
    остальные n, совпадающее с результатом автора tree-qwen3-8b-t1-p5 (6 раундов).
    """
    require_int_at_least("devices", devices, 1)
    return 2 * math.ceil(math.log2(devices))


def tree_allreduce_seconds(
    devices: int, message_bytes: float, bandwidth: float, round_latency: float
) -> float:
    """T_tree = 2·ceil(log2 n)·(α + M/B), формула (6-10).

    Несегментированное биномиальное дерево: в каждом раунде по критическому пути
    идёт полный тензор (tree_collective.py автора).
    """
    require_non_negative("message_bytes", message_bytes)
    require_positive("bandwidth", bandwidth)
    require_non_negative("round_latency", round_latency)
    rounds = tree_allreduce_rounds(devices)
    return rounds * (round_latency + message_bytes / bandwidth)


def ring_tree_crossover_bytes(
    devices: int, bandwidth: float, round_latency: float
) -> float | None:
    """Объём сообщения, при котором (6-9) и (6-10) равны; ниже него дерево быстрее.

    M_* = αB(L − (n − 1)) / ((n − 1)/n − L), L = ceil(log2 n). При n ≤ 3
    точка равна нулю: дерево не быстрее кольца; при n = 1 обе операции пусты.
    """
    require_int_at_least("devices", devices, 1)
    require_positive("bandwidth", bandwidth)
    require_non_negative("round_latency", round_latency)
    if devices == 1:
        return None
    levels = tree_allreduce_rounds(devices) // 2
    numerator = round_latency * bandwidth * (levels - (devices - 1))
    # max: при n = 2 и 3 числитель равен нулю, и частное было бы −0.0
    return max(0.0, numerator / ((devices - 1) / devices - levels))


def tp_step_seconds(
    local_seconds_single: float,
    tp: int,
    reductions: int,
    message_bytes: float,
    bandwidth: float,
    round_latency: float,
) -> float:
    """Шаг при TP: локальная часть делится на tp, добавляются кольцевые редукции (формулы 6-5, 6-9).

    Деление локальной части на tp идеальное: дисбаланс шардов и накладные
    расходы вне коллективных операций не учитываются. При tp = 1 кольцо
    вырождается в ноль раундов, поэтому коммуникация равна нулю.
    """
    require_non_negative("local_seconds_single", local_seconds_single)
    require_int_at_least("tp", tp, 1)
    require_int_at_least("reductions", reductions, 0)
    require_non_negative("message_bytes", message_bytes)
    require_positive("bandwidth", bandwidth)
    require_non_negative("round_latency", round_latency)
    communication = reductions * ring_allreduce_seconds(
        tp, message_bytes, bandwidth, round_latency
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
            require_int_at_least(f"counts[{i}][{j}]", count, 0)
    require_int_at_least("vector_bytes", vector_bytes, 0)
    require_positive("bandwidth", bandwidth)
    require_non_negative("startup_seconds", startup_seconds)
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
