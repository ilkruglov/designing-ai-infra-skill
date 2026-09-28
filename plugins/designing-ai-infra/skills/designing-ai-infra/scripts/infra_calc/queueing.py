"""Нагрузка во времени и очередь (главы 3.1, 8.6).

Средней мощности может хватать, а очередь всё равно растёт, если смесь
классов запросов меняется во времени (пример 3-1).
"""

from __future__ import annotations

from .roofline import _non_negative, _positive


def demand(
    arrivals_per_second: dict[str, float], classes: dict[str, tuple[int, int]]
) -> tuple[float, float]:
    """Новые входные токены в секунду и последующие шаги decode в секунду.

    classes: имя → (входные токены, выходные токены); первый выходной токен
    даёт prefill, поэтому шагов decode на запрос output − 1.
    """
    for name, rate in arrivals_per_second.items():
        _non_negative(f"интенсивность класса {name!r}", rate)
        inputs, outputs = classes[name]
        _non_negative(f"входные токены класса {name!r}", inputs)
        if outputs < 1:
            raise ValueError(
                f"выходных токенов класса {name!r} должно быть не меньше 1: {outputs}"
            )
    tokens = sum(rate * classes[name][0] for name, rate in arrivals_per_second.items())
    steps = sum(
        rate * (classes[name][1] - 1) for name, rate in arrivals_per_second.items()
    )
    return tokens, steps


def utilization(demand_per_second: float, capacity_per_second: float) -> float:
    _non_negative("demand_per_second", demand_per_second)
    _positive("capacity_per_second", capacity_per_second)
    return demand_per_second / capacity_per_second


def littles_law_in_system(arrival_rate: float, time_in_system: float) -> float:
    _non_negative("arrival_rate", arrival_rate)
    _non_negative("time_in_system", time_in_system)
    return arrival_rate * time_in_system
