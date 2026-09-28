"""Единицы измерения (глава 1.2.3).

GB, MB и Mbit — десятичные (10^9, 10^6), GiB и MiB — двоичные (2^30, 2^20).
Число без явной единицы не принимается: путаница GB и GiB или байт и бит
даёт ошибку на порядок, а не на проценты.
"""

from __future__ import annotations

import re
from decimal import Decimal

DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "tf32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp4": 0.5,
    "int4": 0.5,
    "mxfp4": 0.5,
}
UNITS: dict[str, int] = {
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KiB": 2**10,
    "MiB": 2**20,
    "GiB": 2**30,
    "TiB": 2**40,
}
_BYTES = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[KMGT]i?B|B)\s*$")


DTYPE_ALIASES: dict[str, str] = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
}


def dtype_bytes(dtype: str) -> float:
    key = dtype.lower()
    key = DTYPE_ALIASES.get(key, key)
    if key not in DTYPE_BYTES:
        known = ", ".join(sorted(DTYPE_BYTES))
        raise ValueError(f"неизвестный тип данных {dtype!r}; известны: {known}")
    return DTYPE_BYTES[key]


def to_unit(value_bytes: float, unit: str) -> float:
    if unit not in UNITS:
        raise ValueError(f"неизвестная единица {unit!r}; известны: {', '.join(UNITS)}")
    return value_bytes / UNITS[unit]


def parse_bytes(text: str) -> float:
    match = _BYTES.match(text)
    if not match:
        raise ValueError(
            f"объём {text!r} без однозначной единицы; укажите B, KB, MB, GB, TB "
            "или KiB, MiB, GiB, TiB"
        )
    # Decimal: 16.38 GB должно дать ровно 16.38e9, а не 16379999999.999998
    return float(Decimal(match.group("value")) * UNITS[match.group("unit")])


def megabits_per_second_to_bytes(mbps: float) -> float:
    return mbps * 10**6 / 8
