"""Стоимость вызовов модели и выполненной задачи (глава 11.4).

Цены передаются аргументами в долларах за миллион токенов: они меняются
быстрее всего остального и в скилл не встраиваются.
"""

from __future__ import annotations

from .checks import require_int_at_least, require_non_negative


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
    """C = (I_u·p_u + I_w·p_w + I_h·p_h + O·p_o) / 10^6, глава 11.4.2.

    Три категории входа не пересекаются: input_tokens — только обычный вход I_u
    без кэшированных токенов, cache_write_tokens — I_w, cached_tokens — I_h.
    Многие API сообщают prompt_tokens вместе с кэшированными; такое число
    нужно сначала разложить, иначе кэшированные токены оплачиваются дважды.
    output_tokens — O, сумма токенов рассуждения и видимого вывода.
    Хранение кэша и вызовы инструментов (C_storage, C_tool) не входят.
    """
    for name, count in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("cached_tokens", cached_tokens),
        ("cache_write_tokens", cache_write_tokens),
    ):
        require_int_at_least(name, count, 0)
    for name, price in (
        ("input_price", input_price),
        ("output_price", output_price),
        ("cache_read_price", cache_read_price),
        ("cache_write_price", cache_write_price),
    ):
        require_non_negative(name, price)
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
    require_non_negative("cost_per_attempt", cost_per_attempt)
    if not 0 < success_probability <= 1:
        raise ValueError(
            f"вероятность успеха должна лежать в (0, 1]: {success_probability}"
        )
    return cost_per_attempt / success_probability
