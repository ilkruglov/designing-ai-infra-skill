"""Снимок спецификаций ускорителей из calculations/configs/hardware.json оригинала.

Каждое число в снимке имеет источник и локатор. Снимок датирован коммитом
оригинала; для закупки или развёртывания ключевые цифры сверяются с текущей
спецификацией производителя.

- memory_bytes — номинальная ёмкость по этикетке производителя (GB = 10⁹ B),
  а не доступный среде выполнения объём; shared_with_cpu отмечает память,
  общую с CPU и ОС.
- Часть записей — не одна карта, а стойка, сервер или суперчип
  (spec_scope gpu_aggregate / npu_aggregate): ёмкость, пропускная
  способность и пики в них суммированы по device_count устройствам.
  device() выдаёт такую запись только при allow_aggregate=True, чтобы
  стойку нельзя было по ошибке посчитать как одну карту.
- peak_flops() берёт только плотный или structured пик с плавающей точкой;
  по умолчанию — накопление FP32, как в расчётах книги. Записи с
  sparsity «unspecified» и целочисленные TOPS знаменателем не служат.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from .checks import require_int_at_least

DATA = Path(__file__).resolve().parents[2] / "data" / "hardware.json"
# Дата исходного снимка данных; на новом пине содержимое hardware.json не изменилось.
SNAPSHOT = (
    "bojieli/ai-infra-book@d0cc188b (2026-09-26), calculations/configs/hardware.json"
)
_CAPACITY_UNITS = {"GB": 10**9, "GiB": 2**30}
_SCOPE_COUNTS = {
    "single_device": None,
    "gpu_aggregate": "gpu_count",
    "npu_aggregate": "npu_count",
}
_DENOMINATOR_SPARSITY = ("dense", "structured")


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    memory_bytes: int | None
    bandwidth: float | None
    # записи пиков только для чтения: снимок кэшируется load_devices()
    peaks: tuple[Mapping[str, Any], ...]
    scope: str = "single_device"
    device_count: int = 1
    shared_with_cpu: bool = False


def _freeze(value: Any) -> Any:
    """Глубокая копия JSON только для чтения: dict — MappingProxyType, list — tuple."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _parse_device(entry: dict) -> Device:
    device_id = entry["id"]
    memory = entry.get("memory", {})
    unit_name = memory.get("capacity_unit", "GB")
    if unit_name not in _CAPACITY_UNITS:
        raise ValueError(
            f"у {device_id} неизвестная единица ёмкости {unit_name!r}; "
            f"известны: {', '.join(_CAPACITY_UNITS)}"
        )
    scope = entry.get("spec_scope", "single_device")
    if scope not in _SCOPE_COUNTS:
        raise ValueError(
            f"у {device_id} неизвестный spec_scope {scope!r}; "
            f"известны: {', '.join(_SCOPE_COUNTS)}"
        )
    count_field = _SCOPE_COUNTS[scope]
    count = 1
    if count_field is not None:
        raw_count = entry.get(count_field)
        require_int_at_least(f"{count_field} у {device_id}", raw_count, 2)
        count = cast(int, raw_count)  # целое ≥ 2 проверено строкой выше
    capacity = memory.get("nominal_capacity")
    return Device(
        id=device_id,
        name=entry.get("name", device_id),
        memory_bytes=(
            int(capacity * _CAPACITY_UNITS[unit_name]) if capacity is not None else None
        ),
        bandwidth=memory.get("bandwidth_bytes_per_second"),
        peaks=tuple(_freeze(peak) for peak in entry.get("peak_rates", [])),
        scope=scope,
        device_count=count,
        shared_with_cpu=bool(memory.get("shared_with_cpu", False)),
    )


@cache
def load_devices(path: Path = DATA) -> Mapping[str, Device]:
    """Все записи снимка по id; отображение только для чтения, так как кэшируется."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    devices = {}
    for entry in raw["devices"]:
        parsed = _parse_device(entry)
        devices[parsed.id] = parsed
    return MappingProxyType(devices)


def device(device_id: str, allow_aggregate: bool = False) -> Device:
    devices = load_devices()
    if device_id not in devices:
        close = difflib.get_close_matches(device_id, list(devices), n=5, cutoff=0.3)
        raise KeyError(
            f"устройство {device_id!r} не найдено в снимке {SNAPSHOT}; "
            f"похожие: {', '.join(close) or '—'}"
        )
    dev = devices[device_id]
    if dev.scope != "single_device" and not allow_aggregate:
        kind = "GPU" if dev.scope == "gpu_aggregate" else "NPU"
        raise ValueError(
            f"{dev.id} в снимке {SNAPSHOT} — агрегат из {dev.device_count} {kind} "
            f"({dev.scope}): ёмкость, пропускная способность и пики суммированы по "
            f"{dev.device_count} устройствам. Для одной карты выберите одиночное "
            "устройство; агрегат целиком — allow_aggregate=True"
        )
    return dev


def _label(peak: Mapping[str, Any]) -> str:
    return "/".join(
        str(peak.get(field))
        for field in (
            "input_precision",
            "accumulator_precision",
            "execution_unit",
            "sparsity",
        )
    )


def peak_flops(
    dev: Device,
    precision: str = "BF16",
    unit: str = "tensor",
    sparsity: str = "dense",
    accumulator: str | None = "FP32",
) -> float:
    """Пик FLOP/s; accumulator=None разрешает любое накопление, если пик однозначен."""
    if sparsity not in _DENOMINATOR_SPARSITY:
        raise ValueError(
            f"sparsity {sparsity!r} не годится для нижней границы: «unspecified» в "
            f"снимке {SNAPSHOT} не говорит, плотный ли это пик; допустимы: "
            f"{', '.join(_DENOMINATOR_SPARSITY)}"
        )
    same = [
        p
        for p in dev.peaks
        if p.get("input_precision") == precision
        and p.get("execution_unit") == unit
        and p.get("sparsity") == sparsity
        and (accumulator is None or p.get("accumulator_precision") == accumulator)
        and p.get("tera_ops_per_second") is not None
    ]
    matches = [p for p in same if p.get("operation_kind") == "floating_point"]
    if same and not matches:
        raise ValueError(
            f"у {dev.id} пик {precision}/{unit}/{sparsity} в снимке {SNAPSHOT} — "
            "целочисленные TOPS, а не FLOP/s; знаменателем для FLOPs они не служат"
        )
    if not matches:
        available = sorted(
            {
                _label(p)
                for p in dev.peaks
                if p.get("operation_kind") == "floating_point"
                and p.get("sparsity") in _DENOMINATOR_SPARSITY
                and p.get("tera_ops_per_second") is not None
            }
        )
        raise ValueError(
            f"у {dev.id} нет пика {precision}/{accumulator or '*'}/{unit}/{sparsity} "
            f"в снимке {SNAPSHOT}; доступны (вход/накопление/блок/sparsity): "
            f"{', '.join(available) or '—'}; передайте значение аргументом --peak-tflops"
        )
    values = {float(p["tera_ops_per_second"]) for p in matches}
    if len(values) > 1:
        options = ", ".join(
            f"{p.get('accumulator_precision')}: {p['tera_ops_per_second']}"
            for p in matches
        )
        raise ValueError(
            f"у {dev.id} пик {precision}/{unit}/{sparsity} в снимке {SNAPSHOT} "
            f"зависит от накопления ({options} TFLOP/s); укажите accumulator"
        )
    return values.pop() * 1e12


def bandwidth(dev: Device) -> float:
    if dev.bandwidth is None:
        raise ValueError(
            f"у {dev.id} пропускная способность памяти не опубликована в снимке "
            f"{SNAPSHOT}; передайте --bandwidth"
        )
    return float(dev.bandwidth)
