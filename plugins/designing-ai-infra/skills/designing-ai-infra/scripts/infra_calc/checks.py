"""Общие проверки входов калькуляторов.

Отрицательное, нулевое, бесконечное, дробное там, где нужно целое, или NaN
значение даёт правдоподобное, но бессмысленное число, поэтому калькуляторы
отклоняют такие входы с ValueError. Конечность проверяется первой, чтобы NaN
и бесконечность назывались своим именем, а не «отрицательным» значением.
"""

from __future__ import annotations

import math

# Согласование сказуемого с подлежащим: m — «flops должен», f — «интенсивность
# должна», n — «время должно», pl — «токены должны».
_FORMS: dict[str, dict[str, str]] = {
    "m": {
        "must": "должен",
        "cannot": "не может",
        "number": "конечным числом",
        "integer": "целым числом",
        "negative": "отрицательным",
    },
    "f": {
        "must": "должна",
        "cannot": "не может",
        "number": "конечным числом",
        "integer": "целым числом",
        "negative": "отрицательной",
    },
    "n": {
        "must": "должно",
        "cannot": "не может",
        "number": "конечным числом",
        "integer": "целым числом",
        "negative": "отрицательным",
    },
    "pl": {
        "must": "должны",
        "cannot": "не могут",
        "number": "конечными числами",
        "integer": "целыми числами",
        "negative": "отрицательными",
    },
}


def _words(form: str) -> dict[str, str]:
    if form not in _FORMS:
        raise ValueError(
            f"неизвестная грамматическая форма {form!r}; известны: {', '.join(_FORMS)}"
        )
    return _FORMS[form]


def require_finite(name: str, value: float, *, form: str = "m") -> None:
    w = _words(form)
    if not math.isfinite(value):
        raise ValueError(f"{name} {w['must']} быть {w['number']}: {value}")


def require_positive(name: str, value: float, *, form: str = "m") -> None:
    require_finite(name, value, form=form)
    if not value > 0:
        raise ValueError(f"{name} {_words(form)['must']} быть больше нуля: {value}")


def require_non_negative(name: str, value: float, *, form: str = "m") -> None:
    require_finite(name, value, form=form)
    if not value >= 0:
        w = _words(form)
        raise ValueError(f"{name} {w['cannot']} быть {w['negative']}: {value}")


def require_int_at_least(
    name: str, value: object, minimum: int, *, form: str = "m"
) -> None:
    # value: object — проверяется любой вход, в том числе из JSON снимка
    w = _words(form)
    if isinstance(value, bool) or not isinstance(value, int):
        # ValueError, а не TypeError: все отказы входа калькуляторы сообщают одним типом
        raise ValueError(f"{name} {w['must']} быть {w['integer']}: {value!r}")  # noqa: TRY004
    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{name} {w['cannot']} быть {w['negative']}: {value}")
        raise ValueError(f"{name} {w['must']} быть не меньше {minimum}: {value}")
