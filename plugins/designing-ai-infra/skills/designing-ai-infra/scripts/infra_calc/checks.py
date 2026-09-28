"""Общие проверки входов калькуляторов.

Отрицательное, нулевое, бесконечное, дробное там, где нужно целое, или NaN
значение даёт правдоподобное, но бессмысленное число, поэтому калькуляторы
отклоняют такие входы с ValueError. Сравнения записаны как `not value > 0`,
чтобы NaN тоже отклонялся.
"""

from __future__ import annotations

import math


def require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} должен быть конечным числом: {value}")


def require_positive(name: str, value: float) -> None:
    if not value > 0:
        raise ValueError(f"{name} должен быть больше нуля: {value}")
    require_finite(name, value)


def require_non_negative(name: str, value: float) -> None:
    if not value >= 0:
        raise ValueError(f"{name} не может быть отрицательным: {value}")
    require_finite(name, value)


def require_int_at_least(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        # ValueError, а не TypeError: все отказы входа калькуляторы сообщают одним типом
        raise ValueError(f"{name} должен быть целым числом: {value!r}")  # noqa: TRY004
    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{name} не может быть отрицательным: {value}")
        raise ValueError(f"{name} должен быть не меньше {minimum}: {value}")
