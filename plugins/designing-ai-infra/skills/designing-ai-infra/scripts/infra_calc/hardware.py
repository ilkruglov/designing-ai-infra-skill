"""Снимок спецификаций ускорителей из calculations/configs/hardware.json оригинала.

Каждое число в снимке имеет источник и локатор. Снимок датирован коммитом
оригинала; для закупки или развёртывания ключевые цифры сверяются с текущей
спецификацией производителя.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "hardware.json"
SNAPSHOT = "bojieli/ai-infra-book@56ecb425, calculations/configs/hardware.json"
_CAPACITY_UNITS = {"GB": 10**9, "GiB": 2**30}


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    memory_bytes: int | None
    bandwidth: float | None
    peaks: tuple[dict, ...]


@cache
def load_devices(path: Path = DATA) -> dict[str, Device]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    devices: dict[str, Device] = {}
    for entry in raw["devices"]:
        memory = entry.get("memory", {})
        capacity = memory.get("nominal_capacity")
        unit = _CAPACITY_UNITS.get(memory.get("capacity_unit", "GB"), 10**9)
        devices[entry["id"]] = Device(
            id=entry["id"],
            name=entry.get("name", entry["id"]),
            memory_bytes=int(capacity * unit) if capacity is not None else None,
            bandwidth=memory.get("bandwidth_bytes_per_second"),
            peaks=tuple(entry.get("peak_rates", [])),
        )
    return devices


def device(device_id: str) -> Device:
    devices = load_devices()
    if device_id not in devices:
        close = difflib.get_close_matches(device_id, list(devices), n=5, cutoff=0.3)
        raise KeyError(
            f"устройство {device_id!r} не найдено в снимке {SNAPSHOT}; "
            f"похожие: {', '.join(close) or '—'}"
        )
    return devices[device_id]


def peak_flops(
    dev: Device,
    precision: str = "BF16",
    unit: str = "tensor",
    sparsity: str = "dense",
    accumulator: str | None = None,
) -> float:
    matches = [
        p
        for p in dev.peaks
        if p.get("input_precision") == precision
        and p.get("execution_unit") == unit
        and p.get("sparsity") == sparsity
        and (accumulator is None or p.get("accumulator_precision") == accumulator)
        and p.get("tera_ops_per_second") is not None
    ]
    if not matches:
        available = sorted(
            {
                f"{p.get('input_precision')}/{p.get('execution_unit')}/{p.get('sparsity')}"
                for p in dev.peaks
            }
        )
        raise ValueError(
            f"у {dev.id} нет пика {precision}/{unit}/{sparsity} в снимке {SNAPSHOT}; "
            f"доступны: {', '.join(available) or '—'}; "
            "передайте значение аргументом --peak-tflops"
        )
    return max(float(p["tera_ops_per_second"]) for p in matches) * 1e12


def bandwidth(dev: Device) -> float:
    if dev.bandwidth is None:
        raise ValueError(
            f"у {dev.id} пропускная способность памяти не опубликована в снимке; "
            "передайте --bandwidth"
        )
    return float(dev.bandwidth)
