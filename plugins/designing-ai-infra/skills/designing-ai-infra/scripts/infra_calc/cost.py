"""Стоимость вызовов модели и выполненной задачи (глава 11.4).

Цены передаются аргументами в долларах за миллион токенов: они меняются
быстрее всего остального и в скилл не встраиваются.
"""

from __future__ import annotations

from .roofline import _non_negative


def call_cost(
    input_tokens: int,
    output_tokens: int,
    input_price: float,
    output_price: float,
    cached_tokens: int = 0,
    cache_read_price: float = 0,
    cache_write_tokens: int = 0,
    cache_write_price: float = 0,
) -> float:
    for name, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("input_price", input_price),
        ("output_price", output_price),
        ("cached_tokens", cached_tokens),
        ("cache_read_price", cache_read_price),
        ("cache_write_tokens", cache_write_tokens),
        ("cache_write_price", cache_write_price),
    ):
        _non_negative(name, value)
    dollars = (
        input_tokens * input_price
        + output_tokens * output_price
        + cached_tokens * cache_read_price
        + cache_write_tokens * cache_write_price
    )
    return dollars / 1e6


def cost_per_accepted_task(
    cost_per_attempt: float, success_probability: float
) -> float:
    _non_negative("cost_per_attempt", cost_per_attempt)
    if not 0 < success_probability <= 1:
        raise ValueError(
            f"вероятность успеха должна лежать в (0, 1]: {success_probability}"
        )
    return cost_per_attempt / success_probability
