"""CLI калькуляторов. Вывод: JSON или Markdown с формулой, входами и якорем.

Каждый результат несёт формулу, входные данные с единицами, тип границы и
якорь на заголовок книги. Если число взято из снимка железа, вывод называет
снимок; стойка или сервер из снимка берутся только с флагом --allow-aggregate.
Ошибка входа — и разбора аргументов, и проверки значений — завершает команду
с кодом 2 и сообщением на русском, которое называет флаг CLI.

collectives.all_to_all_phase и collectives.tp_step_seconds вызываются из
Python: матрицу назначений неудобно передавать аргументами.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from . import (
    accounting,
    collectives,
    cost,
    edge,
    flops,
    hardware,
    queueing,
    roofline,
    serving,
    speculative,
    training,
    units,
)
from .checks import (
    require_finite,
    require_int_at_least,
    require_non_negative,
    require_positive,
)
from .model import ModelSpec, UnsupportedArchitecture, load_config, parse_spec
from .result import Result, format_number

BOOK = "references/source-book"
A_CH1_KEYS = f"{BOOK}/chapter1.md:152"
A_CH1_UNITS = f"{BOOK}/chapter1.md:189"
A_CH1_TIME = f"{BOOK}/chapter1.md:263"
A_CH1_REF = f"{BOOK}/chapter1.md:450"
A_CH2_MEM = f"{BOOK}/chapter2.md:236"
A_CH2_HYBRID = f"{BOOK}/chapter2.md:418"
A_CH2_EXPERTS = f"{BOOK}/chapter2.md:542"
A_CH3_QUEUE = f"{BOOK}/chapter3.md:75"
A_CH3_PRICE = f"{BOOK}/chapter3.md:209"
A_CH3_STATE = f"{BOOK}/chapter3.md:384"
A_CH3_TRAIN = f"{BOOK}/chapter3.md:442"
A_CH6_TP = f"{BOOK}/chapter6.md:131"
A_CH6_UNION = f"{BOOK}/chapter6.md:391"
A_CH6_RING = f"{BOOK}/chapter6.md:473"
A_CH7_TIME = f"{BOOK}/chapter7.md:938"
A_CH8_MEM = f"{BOOK}/chapter8.md:52"
A_CH8_SHARED = f"{BOOK}/chapter8.md:268"
A_CH8_SPEC = f"{BOOK}/chapter8.md:536"
A_CH8_QUEUE = f"{BOOK}/chapter8.md:598"
A_CH10_ZERO = f"{BOOK}/chapter10.md:151"
A_CH10_PIPE = f"{BOOK}/chapter10.md:322"
A_CH10_CKPT = f"{BOOK}/chapter10.md:553"
A_CH10_SPIKE = f"{BOOK}/chapter10.md:600"
A_CH11_LITTLE = f"{BOOK}/chapter11.md:62"
A_CH11_CALL = f"{BOOK}/chapter11.md:488"
A_CH11_TASK = f"{BOOK}/chapter11.md:515"
A_CH12_EDGE = f"{BOOK}/chapter12.md:11"
QUEUE_ANCHORS = {"ch03": A_CH3_QUEUE, "ch08": A_CH8_QUEUE, "ch11": A_CH11_LITTLE}

SINGLE = "single_device"
USER = "задано пользователем"
BF16_BYTES = 2.0
ZERO_FORMULAS = (
    "P × (2 + 2 + 12)",
    "P × (2 + 2 + 12/N)",
    "P × (2 + (2 + 12)/N)",
    "P × (2 + 2 + 12) / N",
)
QUANT_NOTE = (
    "квантизованные веса: parameters × bytes не учитывает scale групп и части "
    "модели, оставленные в BF16; для 8-битной DeepSeek-R1-Distill-Llama-70B книга "
    "даёт 73.73 GB против 70.55 GB по этой формуле (глава 1.2.3); известные "
    "накладные расходы передайте через --quant-overhead-bytes"
)
QUANT_READ_NOTE = (
    "квантизованные веса: scale групп и части модели в BF16 в чтение не включены"
)
RATES_SUFFIX = "; граница предполагает идеальное деление работы между устройствами"

# Единица каждого входа результата. Пустая строка у числа допустима только для
# безразмерных величин из DIMENSIONLESS; у строк и флагов это метка.
DIMENSIONLESS = frozenset(
    {
        "params",
        "parameters",
        "active_parameters",
        "device_count",
        "batch",
        "dp",
        "stage",
        "devices",
        "mfu",
        "stages",
        "microbatches",
        "virtual_stages",
        "tp",
        "experts",
        "top_k",
        "experts_per_layer",
        "acceptance",
        "calls",
        "success",
    }
)
LABELS = frozenset(
    {
        "config",
        "model_type",
        "wrapper_model_type",
        "weight_dtype",
        "kv_dtype",
        "device",
        "name",
        "scope",
        "shared_with_cpu",
        "precision",
        "accumulator",
        "sparsity",
        "peak_source",
        "bandwidth_source",
        "memory_source",
        "size_text",
        "dtype",
        "classes",
        "rates",
        "rounds",
        "price_scope",
    }
)
INPUT_UNITS: dict[str, str] = {
    **dict.fromkeys(DIMENSIONLESS | LABELS, ""),
    "context": "tok",
    "memory_context": "tok",
    "shared_prefix": "tok",
    "window": "tok",
    "fixed_state": "B",
    "tokens": "tok",
    "total_tokens": "tok",
    "draft": "tok",
    "input": "tok",
    "output": "tok",
    "cached": "tok",
    "cache_write": "tok",
    "flops": "FLOP",
    "decode_flops": "FLOP",
    "prefill_flops": "FLOP",
    "total_flops": "FLOP",
    "bytes": "B",
    "weights": "B",
    "weight_read": "B",
    "memory": "B",
    "reserve": "B",
    "kv_request": "B",
    "kv_request_per_device": "B",
    "kv_shared_prefix": "B",
    "message": "B",
    "upload": "B",
    "download": "B",
    "size_bytes": "B",
    "quant_overhead": "B",
    "kv_per_token": "B/tok",
    "peak": "FLOP/s",
    "bandwidth": "B/s",
    "tokens_per_second": "tok/s",
    "prefill_capacity": "tok/s",
    "input_tokens_per_second": "tok/s",
    "decode_capacity": "step/s",
    "decode_steps_per_second": "step/s",
    "price_per_hour": "$/h",
    "cost_per_call": "$",
    "input_price": "$/Mtok",
    "output_price": "$/Mtok",
    "cache_read_price": "$/Mtok",
    "cache_write_price": "$/Mtok",
    "save_seconds": "s",
    "recovery": "s",
    "interval": "s",
    "plain_step": "s",
    "alpha": "s",
    "rtt": "s",
    "compute": "s",
    "time_in_system": "s",
    "rounds_seconds": "s",
    "forward_seconds": "s",
    "backward_seconds": "s",
    "common_job_mtbf": "s",
    "weight_bytes_per_param": "B",
    "grad_bytes_per_param": "B",
    "optimizer_bytes_per_param": "B",
    "failure_rate": "1/s",
    "arrival_rate": "1/s",
    "mbps": "Mbit/s",
    "up_mbps": "Mbit/s",
    "down_mbps": "Mbit/s",
}


def _result(
    name: str,
    value: float | None,
    unit: str,
    formula: str,
    inputs: dict[str, Any],
    anchor: str,
    *,
    bound: str | None = None,
    notes: Iterable[str] = (),
) -> Result:
    unknown = sorted(key for key in inputs if key not in INPUT_UNITS)
    if unknown:
        # ошибка программы, а не входа: не превращается в код 2
        raise RuntimeError(f"нет единицы для входов {unknown} результата {name}")
    return Result(
        name,
        value,
        unit,
        formula,
        inputs,
        anchor,
        bound=bound,
        notes=tuple(notes),
        input_units={key: INPUT_UNITS[key] for key in inputs},
    )


# --- проверка входов по флагам ----------------------------------------------


def _attribute(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def _check(
    args: argparse.Namespace,
    *,
    positive: Sequence[str] = (),
    non_negative: Sequence[str] = (),
    at_least: Sequence[tuple[str, int]] = (),
) -> None:
    """Проверить заданные флаги до вызова калькулятора, чтобы ошибка называла флаг."""
    for flag in positive:
        value = getattr(args, _attribute(flag))
        if value is not None:
            require_positive(flag, value)
    for flag in non_negative:
        value = getattr(args, _attribute(flag))
        if value is not None:
            require_non_negative(flag, value)
    for flag, minimum in at_least:
        value = getattr(args, _attribute(flag))
        if value is not None:
            require_int_at_least(flag, value, minimum)


def _check_fraction(flag: str, value: float | None, zero_allowed: bool) -> None:
    if value is None:
        return
    require_finite(flag, value)
    if not 0 <= value <= 1 or (value == 0 and not zero_allowed):
        interval = "[0, 1]" if zero_allowed else "(0, 1]"
        raise ValueError(f"{flag} должен лежать в {interval}: {value}")


# --- общие части ------------------------------------------------------------


def _load_spec(path: str) -> ModelSpec:
    """config.json → ModelSpec; ошибки чтения и разбора — ValueError на русском."""
    try:
        config = load_config(path)
    except FileNotFoundError:
        raise ValueError(f"файл конфигурации не найден: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path} — не JSON: {error.msg}, строка {error.lineno}"
        ) from None
    except OSError as error:
        raise ValueError(f"не удалось прочитать {path}: {error.strerror}") from None
    if not isinstance(config, dict):
        # ValueError, а не TypeError: все отказы входа CLI сообщает одним типом
        raise ValueError(f"{path}: config.json должен быть объектом JSON")  # noqa: TRY004
    try:
        return parse_spec(config)
    except UnsupportedArchitecture:
        raise
    except KeyError as error:
        raise ValueError(f"в {path} нет обязательного поля {error.args[0]!r}") from None
    except (TypeError, ValueError) as error:
        raise ValueError(f"в {path} поле неверного типа: {error}") from None


def _wrapper_notes(spec: ModelSpec) -> tuple[str, ...]:
    if spec.wrapper_model_type is None:
        return ()
    note = (
        f"config — обёртка {spec.wrapper_model_type} над текстовой моделью "
        f"{spec.model_type}: веса энкодеров (vision/audio) и проектора не учтены"
    )
    return (note,)


def _aggregate_note(
    dev: hardware.Device | None, labels: Sequence[str], suffix: str = ""
) -> tuple[str, ...]:
    """Примечание об агрегате: только для величин, взятых из снимка."""
    if dev is None or dev.scope == SINGLE or not labels:
        return ()
    n = dev.device_count
    return (
        (
            f"{dev.id} — агрегат {dev.scope} из {n} устройств; сумма по {n} "
            f"устройствам, а не значение одной карты: {', '.join(labels)}{suffix}"
        ),
    )


def _device(args: argparse.Namespace, per_device: str) -> hardware.Device | None:
    if not args.device:
        return None
    dev = hardware.device(args.device, allow_aggregate=True)
    if dev.scope != SINGLE and not args.allow_aggregate:
        raise ValueError(
            f"{dev.id} в снимке {hardware.SNAPSHOT} — агрегат {dev.scope} из "
            f"{dev.device_count} устройств: ёмкость, пропускная способность и пики "
            f"суммированы по {dev.device_count} устройствам. Для одной карты выберите "
            "одиночное устройство или задайте значения на одно устройство без "
            f"--device: {per_device}; агрегат целиком — флаг --allow-aggregate"
        )
    return dev


def _snapshot_peak(dev: hardware.Device, args: argparse.Namespace) -> float:
    try:
        return hardware.peak_flops(
            dev, args.precision, sparsity=args.sparsity, accumulator=args.accumulator
        )
    except ValueError as error:
        raise ValueError(
            f"{error}. Пик выбирается из снимка по --precision {args.precision}, "
            f"--accumulator {args.accumulator} и --sparsity {args.sparsity}: "
            "укажите накопление из списка доступных через --accumulator "
            "или задайте пик явно через --peak-tflops"
        ) from None


def _refuse_mixed_rates(
    dev: hardware.Device | None, peak: _Quantity, bw: _Quantity
) -> None:
    """У агрегата пик и полоса берутся либо оба из снимка, либо оба явно.

    Значение агрегата в снимке — сумма по всем его устройствам; явное значение
    чаще задают на карту. Одна граница с пиком карты и полосой стойки (или
    наоборот) смешала бы охваты, и понять, какой из них имелся в виду, нельзя.
    """
    if dev is None or dev.scope == SINGLE:
        return
    sources = {"--peak-tflops": peak, "--bandwidth": bw}
    given = [flag for flag, q in sources.items() if q.snapshot_label is None]
    taken = [q.snapshot_label for q in sources.values() if q.snapshot_label]
    if given and taken:
        raise ValueError(
            f"{dev.id} — агрегат {dev.scope} (устройств в агрегате: {dev.device_count}): "
            f"{given[0]} задан явно, а {taken[0]} берётся из снимка как сумма по "
            f"{dev.device_count} устройствам: граница смешала бы величины разного "
            "охвата. Задайте --peak-tflops и --bandwidth вместе или не задавайте ни "
            "один из них"
        )


@dataclass(frozen=True)
class _Quantity:
    """Величина для расчёта: значение, входы для отчёта и откуда она взята."""

    value: float
    inputs: dict[str, Any]
    snapshot_label: str | None = None
    notes: tuple[str, ...] = ()


def _labels(*quantities: _Quantity) -> list[str]:
    return [q.snapshot_label for q in quantities if q.snapshot_label]


def _peak(args: argparse.Namespace, dev: hardware.Device | None) -> _Quantity:
    if args.peak_tflops is not None:
        require_positive("--peak-tflops", args.peak_tflops)
        peak = args.peak_tflops * 1e12
        return _Quantity(peak, {"peak": peak, "peak_source": USER})
    if dev is None:
        raise ValueError("укажите --device или --peak-tflops")
    peak = _snapshot_peak(dev, args)
    inputs = {
        "device": dev.id,
        "precision": args.precision,
        "accumulator": args.accumulator,
        "sparsity": args.sparsity,
        "peak": peak,
    }
    return _Quantity(peak, inputs, "пик")


def _bandwidth(args: argparse.Namespace, dev: hardware.Device | None) -> _Quantity:
    if args.bandwidth is not None:
        require_positive("--bandwidth", args.bandwidth)
        return _Quantity(
            args.bandwidth, {"bandwidth": args.bandwidth, "bandwidth_source": USER}
        )
    if dev is None:
        raise ValueError("укажите --device или --bandwidth")
    value = hardware.bandwidth(dev)
    return _Quantity(
        value, {"device": dev.id, "bandwidth": value}, "пропускная способность памяти"
    )


def _memory(args: argparse.Namespace, dev: hardware.Device | None) -> _Quantity:
    if args.memory is not None:
        return _Quantity(args.memory, {"memory": args.memory, "memory_source": USER})
    if dev is None:
        raise ValueError(
            "укажите --memory (байт памяти одного устройства) или --device"
        )
    if dev.memory_bytes is None:
        raise ValueError(
            f"у {dev.id} ёмкость памяти не опубликована в снимке {hardware.SNAPSHOT}; "
            "передайте --memory"
        )
    notes = [
        (
            "номинальная ёмкость по этикетке, а не доступный среде выполнения объём; "
            "резерв среды выполнения задайте через --reserve"
        )
    ]
    if dev.shared_with_cpu:
        notes.append("память общая с CPU и ОС: модели доступна только её часть")
    return _Quantity(
        dev.memory_bytes,
        {"memory": dev.memory_bytes, "memory_source": dev.id},
        "ёмкость памяти",
        tuple(notes),
    )


# --- команды -------------------------------------------------------------------


def _weight_estimate(
    args: argparse.Namespace, bytes_per_param: float
) -> tuple[int, str | None, tuple[str, ...]]:
    """Добавка к «параметры × байты», тип границы и примечания для квантизации."""
    quantized = bytes_per_param < BF16_BYTES
    overhead = args.quant_overhead_bytes
    if overhead is not None and not quantized:
        raise ValueError(
            "--quant-overhead-bytes задаёт scale и части модели в высокой точности "
            f"квантизованных весов; с --weight-dtype {args.weight_dtype} он не нужен"
        )
    if not quantized:
        return 0, None, ()
    if overhead is None:
        return 0, "lower", (QUANT_NOTE,)
    note = f"прибавлены накладные расходы квантизации из --quant-overhead-bytes: {overhead:.6g} B"
    return int(overhead), None, (note,)


def _kv_tp_note(spec: ModelSpec) -> str:
    """Делится ли KV по TP у этой архитектуры (глава 6.2.2)."""
    if spec.family == "mla_moe":
        return (
            "вывод из глав 2.3.2 и 6.2.2, числового примера TP для MLA в книге нет: "
            "латентный KV MLA по TP не делится — одна скрытая переменная на токен "
            "для всех голов, а TP делит внимание по головам, поэтому каждая карта TP "
            "хранит её целиком"
        )
    note = (
        f"при TP KV делится по головам KV: на карту kv_bytes_per_token / "
        f"min(TP, {spec.kv_heads}); при TP > {spec.kv_heads} головы KV дублируются"
    )
    if spec.family == "hybrid_linear":
        note += "; фиксированное состояние линейных слоёв делится на TP"
    return note


def _model_tp(
    spec: ModelSpec,
    args: argparse.Namespace,
    kb: float,
    weight_bytes: int | None,
    base: dict[str, Any],
    weight_bound: str | None = None,
    weight_notes: tuple[str, ...] = (),
) -> list[Result]:
    tp = args.tp
    kv_div = accounting.tp_kv_divisor(spec, tp)
    inputs = base | {"tp": tp, "kv_dtype": args.kv_dtype}
    kv_notes = [_kv_tp_note(spec)]
    if spec.family != "mla_moe" and tp > spec.kv_heads:
        kv_notes.append(
            f"TP={tp} больше числа голов KV ({spec.kv_heads}): KV дублируется, копий "
            f"каждой головы KV: {tp // spec.kv_heads}"
        )
    kv_formula = (
        "kv_bytes_per_token (не делится)"
        if kv_div == 1
        else f"kv_bytes_per_token / min(TP, {spec.kv_heads})"
    )
    out: list[Result] = []
    if weight_bytes is not None:
        out.append(
            _result(
                "weight_bytes_per_device",
                weight_bytes // tp,
                "B",
                "weight_bytes / TP",
                base | {"tp": tp, "weight_dtype": args.weight_dtype},
                A_CH6_TP,
                bound=weight_bound,
                notes=(
                    (
                        "идеальное деление: матрицы по головам и промежуточному "
                        "измерению, эмбеддинги и голова по словарю; нормализации "
                        "копируются"
                    ),
                    *weight_notes,
                ),
            )
        )
    out.append(
        _result(
            "kv_bytes_per_token_per_device",
            accounting.kv_bytes_per_token(spec, kb) // kv_div,
            "B",
            kv_formula,
            inputs,
            A_CH6_TP,
            notes=kv_notes,
        )
    )
    if args.context:
        out.append(
            _result(
                "kv_resident_bytes_per_device",
                accounting.kv_resident_bytes(spec, args.context, kb) // kv_div,
                "B",
                kv_formula.replace("kv_bytes_per_token", "kv_resident_bytes"),
                inputs | {"context": args.context},
                A_CH6_TP,
                notes=kv_notes,
            )
        )
    recurrent, conv = accounting.fixed_state_bytes(spec)
    if recurrent or conv:
        out.append(
            _result(
                "fixed_state_bytes_per_device",
                (recurrent + conv) // accounting.tp_fixed_state_divisor(spec, tp),
                "B",
                "fixed_state_bytes / TP",
                base | {"tp": tp},
                A_CH6_TP,
                notes=(
                    (
                        "состояние линейных слоёв делится вместе с линейными головами "
                        "(таблица главы 6.2.2: 31/117 запросов Qwen3.6 на двух H100)"
                    ),
                ),
            )
        )
    return out


def _model_batch(
    spec: ModelSpec, args: argparse.Namespace, wb: float, base: dict[str, Any]
) -> list[Result]:
    if args.batch is None:
        raise ValueError(
            "--experts-per-layer задаётся вместе с --batch: число различных "
            "экспертов зависит от батча"
        )
    if not spec.experts:
        raise ValueError(
            "--batch и --experts-per-layer относятся к MoE: у "
            f"{spec.model_type} чтение весов за шаг не зависит от батча"
        )
    batch, k, experts = args.batch, spec.experts_per_token, spec.experts
    inputs = base | {"batch": batch, "experts": experts, "top_k": k}
    if args.experts_per_layer is None:
        union = accounting.expected_active_experts(experts, k, batch)
        union_formula = "E·[1 − (1 − k/E)^B]"
        union_notes: tuple[str, ...] = (
            (
                "равномерная независимая маршрутизация (формула 6-8); decode — один "
                "токен на запрос. Реальная маршрутизация неравномерна: от k (одни и "
                f"те же эксперты) до min(B·k, E) = {min(batch * k, experts)}; "
                "измеренное или худшее значение задайте через --experts-per-layer"
            ),
        )
    else:
        union = args.experts_per_layer
        highest = min(batch * k, experts)
        require_finite("--experts-per-layer", union)
        if not k <= union <= highest:
            raise ValueError(
                "--experts-per-layer должно лежать в [k, min(B·k, E)] = "
                f"[{k}, {highest}]: {union:g}"
            )
        union_formula = f"{USER} (--experts-per-layer)"
        union_notes = ("худший случай для батча B — min(B·k, E)",)
    out = [
        _result(
            "experts_per_layer_at_batch",
            union,
            "",
            union_formula,
            inputs,
            A_CH6_UNION,
            notes=union_notes,
        )
    ]
    try:
        read: int | None = accounting.batch_decode_weight_read_bytes(spec, union, wb)
        read_notes: tuple[str, ...] = (
            "общие веса читаются один раз на шаг, каждый выбранный эксперт — один раз",
        )
    except UnsupportedArchitecture as error:
        read, read_notes = None, (str(error),)
    out.append(
        _result(
            "decode_weight_read_bytes_at_batch",
            read,
            "B",
            "R_общие + U × L_MoE × 3·h·f_e × bytes_per_param"
            if read is not None
            else "не вычисляется",
            inputs | {"experts_per_layer": union, "weight_dtype": args.weight_dtype},
            A_CH2_EXPERTS,
            bound="lower" if read is not None and wb < BF16_BYTES else None,
            notes=read_notes,
        )
    )
    return out


def _model(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        non_negative=("--quant-overhead-bytes",),
        at_least=(("--context", 0), ("--tp", 1), ("--batch", 1)),
    )
    spec = _load_spec(args.config)
    if args.tp is not None:
        accounting.tp_kv_divisor(spec, args.tp)  # отказ до расчётов
    wb = units.dtype_bytes(args.weight_dtype)
    kb = units.dtype_bytes(args.kv_dtype)
    wrapper = _wrapper_notes(spec)
    overhead, weight_bound, weight_notes = _weight_estimate(args, wb)
    base: dict[str, Any] = {
        "config": Path(args.config).name,
        "model_type": spec.model_type,
    }
    if spec.wrapper_model_type:
        base["wrapper_model_type"] = spec.wrapper_model_type
    weights = {**base, "weight_dtype": args.weight_dtype}
    if args.quant_overhead_bytes is not None:
        weights["quant_overhead"] = args.quant_overhead_bytes
    out: list[Result] = []
    try:
        params: int | None = accounting.parameter_count(spec)
        refusal: UnsupportedArchitecture | None = None
    except UnsupportedArchitecture as error:
        params, refusal = None, error

    if params is not None:
        if args.params is not None:
            raise ValueError(
                "--params нужен только для архитектур, где калькулятор не считает "
                f"параметры; для {spec.model_type} по config.json получено "
                f"{format_number(params)}"
            )
        read_notes = wrapper
        if spec.experts:
            read_notes += (
                "эксперты одного токена; при батче читается объединение экспертов",
            )
        if weight_bound == "lower" or overhead:
            read_notes += (QUANT_READ_NOTE,)
        out += [
            _result(
                "parameters",
                params,
                "",
                "Σ матриц + нормализации + эмбеддинги (без MTP)",
                base,
                A_CH3_STATE,
                notes=wrapper,
            ),
            _result(
                "active_parameters",
                accounting.parameter_count(spec, active=True),
                "",
                "эксперты одного токена вместо всех",
                base,
                A_CH2_MEM,
                notes=wrapper,
            ),
            _result(
                "weight_bytes",
                accounting.weight_bytes(spec, wb) + overhead,
                "B",
                "parameters × bytes_per_param"
                + (" + quant_overhead" if overhead else ""),
                weights,
                A_CH1_UNITS,
                bound=weight_bound,
                notes=(*wrapper, *weight_notes),
            ),
            _result(
                "decode_weight_read_bytes",
                accounting.decode_weight_read_bytes(spec, wb),
                "B",
                "(active_parameters − таблица эмбеддингов без общих весов) × bytes_per_param",
                {k: v for k, v in weights.items() if k != "quant_overhead"},
                A_CH2_MEM,
                bound="lower" if wb < BF16_BYTES else None,
                notes=read_notes,
            ),
        ]
    elif args.params is not None and refusal is not None:
        given = {**base, "params": args.params}
        param_notes: tuple[str, ...] = (
            (
                f"калькулятор не считает параметры {refusal.model_type} "
                f"(поля: {', '.join(refusal.fields) or '—'}); значение взято из --params как есть"
            ),
        )
        if spec.wrapper_model_type:
            param_notes += (
                (
                    "включает ли число веса энкодеров обёртки "
                    f"{spec.wrapper_model_type}, определяет карточка модели"
                ),
            )
        out += [
            _result(
                "parameters",
                args.params,
                "",
                f"{USER} (--params)",
                given,
                A_CH3_STATE,
                notes=param_notes,
            ),
            _result(
                "weight_bytes",
                int(args.params * wb) + overhead,
                "B",
                f"parameters ({USER}) × bytes_per_param"
                + (" + quant_overhead" if overhead else ""),
                given | {k: v for k, v in weights.items() if k not in base},
                A_CH1_UNITS,
                bound=weight_bound,
                notes=(*param_notes[1:], *weight_notes),
            ),
        ]
    else:
        out.append(
            _result(
                "parameters",
                None,
                "",
                "не вычисляется",
                base,
                A_CH3_STATE,
                notes=(str(refusal), *wrapper),
            )
        )

    kv_formula = {
        "mla_moe": "layers × (kv_lora_rank + qk_rope_dim) × bytes",
        "hybrid_linear": "2 × full_attention_layers × kv_heads × head_dim × bytes",
    }.get(spec.family, "2 × layers × kv_heads × head_dim × bytes")
    kv_inputs = {**base, "kv_dtype": args.kv_dtype}
    out.append(
        _result(
            "kv_bytes_per_token",
            accounting.kv_bytes_per_token(spec, kb),
            "B",
            kv_formula,
            kv_inputs,
            A_CH8_MEM,
            notes=(_kv_tp_note(spec),),
        )
    )
    if args.context:
        resident = accounting.kv_resident_bytes(spec, args.context, kb)
        window = f"min(context, {spec.window})" if spec.window else "context"
        out.append(
            _result(
                "kv_resident_bytes",
                resident,
                "B",
                f"kv_bytes_per_token × {window}",
                kv_inputs | {"context": args.context},
                A_CH8_MEM,
            )
        )
    recurrent, conv = accounting.fixed_state_bytes(spec)
    if recurrent or conv:
        out.append(
            _result(
                "fixed_state_bytes",
                recurrent + conv,
                "B",
                "линейные слои: рекуррентное (FP32) + свёрточное (BF16) состояние на запрос",
                base,
                A_CH2_HYBRID,
            )
        )
    if args.context:
        out += _forward_flops(spec, args.context, base)
    if args.tp is not None:
        if params is not None:
            weight_total: int | None = accounting.weight_bytes(spec, wb) + overhead
        elif args.params is not None:
            weight_total = int(args.params * wb) + overhead
        else:
            weight_total = None
        out += _model_tp(spec, args, kb, weight_total, base, weight_bound, weight_notes)
    if args.batch is not None or args.experts_per_layer is not None:
        out += _model_batch(spec, args, wb, base)
    return out


def _forward_flops(spec: ModelSpec, context: int, base: dict[str, Any]) -> list[Result]:
    try:
        prefill = flops.forward_matrix_flops(spec, context)
        decode = flops.forward_matrix_flops(spec, 1, context - 1)
    except UnsupportedArchitecture as error:
        return [
            _result(
                "prefill_flops",
                None,
                "FLOP",
                "не вычисляется",
                base,
                A_CH1_REF,
                notes=(str(error),),
            )
        ]
    return [
        _result(
            "prefill_flops",
            prefill,
            "FLOP",
            "2·W_active·n + 2·V·h + 4·L·H·d·n(n+1)/2",
            base | {"tokens": context},
            A_CH1_REF,
        ),
        _result(
            "decode_step_flops",
            decode,
            "FLOP",
            "2·W_active + 2·V·h + 4·L·H·d·context",
            base | {"context": context},
            A_CH1_REF,
        ),
    ]


def _roofline_memory_only(
    args: argparse.Namespace, dev: hardware.Device | None
) -> list[Result]:
    if args.peak_tflops is not None:
        raise ValueError(
            "--memory-only оценивает только чтение; --peak-tflops с ним не задаётся"
        )
    bw = _bandwidth(args, dev)
    notes = _aggregate_note(dev, _labels(bw), RATES_SUFFIX)
    skipped = (
        "вычислительная граница не оценивалась (--memory-only): пик не задан, "
        "граница шага — только по чтению"
    )
    inputs = {"flops": args.flops, "bytes": args.bytes, **bw.inputs}
    memory = roofline.memory_seconds(args.bytes, bw.value)
    out = [
        _result(
            "step_lower_bound_seconds",
            memory,
            "s",
            "R/β (F/Π не оценивался)",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=(skipped, *notes),
        ),
        _result(
            "compute_seconds",
            None,
            "s",
            "не вычисляется",
            inputs,
            A_CH1_TIME,
            notes=(skipped,),
        ),
        _result(
            "memory_seconds",
            memory,
            "s",
            "R/β",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=notes,
        ),
    ]
    if args.bytes > 0:
        out.append(
            _result(
                "arithmetic_intensity",
                roofline.arithmetic_intensity(args.flops, args.bytes),
                "FLOP/B",
                "F/R",
                inputs,
                A_CH1_TIME,
            )
        )
    out.append(
        _result(
            "ridge_point",
            None,
            "FLOP/B",
            "не вычисляется",
            inputs,
            A_CH1_TIME,
            notes=(skipped,),
        )
    )
    return out


def _roofline(args: argparse.Namespace) -> list[Result]:
    _check(args, non_negative=("--flops", "--bytes"))
    dev = _device(args, "--peak-tflops и --bandwidth")
    if args.memory_only:
        return _roofline_memory_only(args, dev)
    try:
        peak = _peak(args, dev)
    except ValueError as error:
        raise ValueError(
            f"{error}; только граница по чтению без пика — флаг --memory-only"
        ) from None
    bw = _bandwidth(args, dev)
    _refuse_mixed_rates(dev, peak, bw)
    notes = _aggregate_note(dev, _labels(peak, bw), RATES_SUFFIX)
    inputs = {"flops": args.flops, "bytes": args.bytes, **peak.inputs, **bw.inputs}
    out = [
        _result(
            "step_lower_bound_seconds",
            roofline.lower_bound_seconds(args.flops, args.bytes, peak.value, bw.value),
            "s",
            "max(F/Π, R/β)",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=notes,
        ),
        _result(
            "compute_seconds",
            roofline.compute_seconds(args.flops, peak.value),
            "s",
            "F/Π",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=notes,
        ),
        _result(
            "memory_seconds",
            roofline.memory_seconds(args.bytes, bw.value),
            "s",
            "R/β",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=notes,
        ),
    ]
    if args.bytes > 0:
        out.append(
            _result(
                "arithmetic_intensity",
                roofline.arithmetic_intensity(args.flops, args.bytes),
                "FLOP/B",
                "F/R",
                inputs,
                A_CH1_TIME,
            )
        )
    out.append(
        _result(
            "ridge_point",
            roofline.ridge_point(peak.value, bw.value),
            "FLOP/B",
            "Π/β",
            inputs,
            A_CH1_TIME,
            notes=notes,
        )
    )
    return out


def _window_split(length: int, prefix: int, window: int | None) -> tuple[int, int]:
    """Токены KV запроса длины length: (собственные, общего префикса).

    Окно внимания оставляет последние min(length, window) токенов; собственные
    — min(length − prefix, window), остальные в окне — хвост общего префикса.
    """
    if not window:
        return length - prefix, prefix
    own = min(length - prefix, window)
    return own, min(length, window) - own


def _window_notes(window: int | None, length: int, name: str) -> tuple[str, ...]:
    """Примечание об окне внимания, когда окно короче длины."""
    if not window or window >= length:
        return ()
    head = (
        f"окно внимания window = {window} ток. (sliding_window из --config): "
        f"KV запроса — последние min({name}, window) = {window} из {length} ток."
    )
    if name == "context":
        tail = (
            "; нижняя граница шага при любом движке. Движок, который держит и "
            "читает KV всего контекста, даёт шаг больше"
        )
    else:
        tail = (
            "; верхняя граница ёмкости. Движок, который держит KV всего контекста, "
            "вмещает меньше запросов"
        )
    return (head + tail,)


def _optional_memory(
    args: argparse.Namespace, dev: hardware.Device | None
) -> tuple[_Quantity | None, str]:
    """Ёмкость для бюджета памяти или причина, почему он не оценивается."""
    if args.memory is None and dev is None:
        return None, (
            "ёмкость памяти не оценивалась: не задан --memory и нет --device; "
            "посчитаны только границы времени"
        )
    if args.memory is None and dev is not None and dev.memory_bytes is None:
        return None, (
            f"ёмкость памяти не оценивалась: у {dev.id} ёмкость не опубликована в "
            f"снимке {hardware.SNAPSHOT}; передайте --memory"
        )
    return _memory(args, dev), ""


def _serving_state(
    args: argparse.Namespace, spec: ModelSpec | None
) -> tuple[float, int, list[str]]:
    """Фиксированное состояние запроса (всей модели), его делитель по TP и примечания."""
    auto = 0
    if spec is not None:
        recurrent, conv = accounting.fixed_state_bytes(spec)
        auto = recurrent + conv
    notes: list[str] = []
    if args.fixed_state_bytes is not None:
        total: float = args.fixed_state_bytes
        if auto:
            notes.append(
                "--fixed-state-bytes заменяет состояние из config: "
                f"{format_number(auto)} B на запрос"
            )
    elif auto:
        total = auto
        notes.append(
            f"фиксированное состояние из config ({spec.model_type if spec else ''}): "
            f"{format_number(auto)} B на запрос — рекуррентное и свёрточное состояние "
            "линейных слоёв"
        )
    else:
        total = 0
    divisor = 1
    if spec is not None and args.tp > 1 and total:
        if spec.family != "hybrid_linear":
            raise ValueError(
                f"как делится --fixed-state-bytes по TP у {spec.model_type}, "
                "калькулятор не знает; задайте состояние на карту без --tp"
            )
        divisor = accounting.tp_fixed_state_divisor(spec, args.tp)
    return total, divisor, notes


def _serving_price(
    args: argparse.Namespace,
    dev: hardware.Device | None,
    aggregate_rates: bool,
    tp: int,
    throughput: float,
    parallel: dict[str, Any],
) -> Result:
    """Цена миллиона токенов; у агрегата область цены задаётся явно.

    Агрегатом цена считается, только когда пик и полоса взяты из записи агрегата:
    тогда токены в секунду — скорость всей записи. С явными --peak-tflops и
    --bandwidth скорость описывает заданное устройство, и цена берётся как есть.
    """
    on_aggregate = dev is not None and dev.scope != SINGLE
    aggregate = on_aggregate and aggregate_rates
    scope = args.price_scope
    if scope == "aggregate" and not aggregate:
        if on_aggregate:
            assert dev is not None
            raise ValueError(
                f"--price-scope aggregate относится к скорости всей записи {dev.id}, а "
                "--peak-tflops и --bandwidth заданы явно: токены в секунду описывают "
                "заданное устройство, и цена за его час берётся как есть — уберите "
                "--price-scope или укажите card"
            )
        raise ValueError(
            "--price-scope aggregate — только для агрегата из снимка (--device … "
            "--allow-aggregate): у одиночного устройства цена — за карту, при "
            "--tp — за все карты экземпляра"
        )
    if aggregate and scope is None:
        assert dev is not None
        raise ValueError(
            f"{dev.id} — агрегат (устройств в агрегате: {dev.device_count}): "
            "неясно, за что --price-per-hour — за час одной карты или всей записи. "
            "Укажите --price-scope card (цена карты × число устройств агрегата) "
            "или --price-scope aggregate (цена всей записи как есть)"
        )
    scope = scope or "card"
    cards = dev.device_count if aggregate and scope == "card" and dev else 1
    multiplier = tp * cards
    if cards > 1:
        formula = f"{cards} × price_per_hour / 3600 / tok_s × 10^6"
        notes: tuple[str, ...] = (f"цена за карту × устройств в агрегате ({cards})",)
    elif tp > 1:
        formula = "TP × price_per_hour / 3600 / tok_s × 10^6"
        notes = (f"цена — за все карты экземпляра, TP = {tp}",)
    else:
        formula = "price_per_hour / 3600 / tok_s × 10^6"
        if aggregate:
            notes = ("цена всей записи агрегата как есть",)
        elif on_aggregate:
            notes = (
                (
                    "--peak-tflops и --bandwidth заданы явно: цена — за час устройства, "
                    "которое они описывают, без умножения на число устройств агрегата"
                ),
            )
        else:
            notes = ()
    return _result(
        "cost_per_million_tokens_lower_bound",
        serving.cost_per_million_tokens(args.price_per_hour * multiplier, throughput),
        "$",
        formula,
        {
            "price_per_hour": args.price_per_hour,
            "price_scope": scope,
            "tokens_per_second": throughput,
            **parallel,
        },
        A_CH3_PRICE,
        bound="lower",
        notes=notes,
    )


def _serving(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--kv-per-token",),
        non_negative=(
            "--weights",
            "--weight-read",
            "--decode-flops",
            "--prefill-flops",
            "--memory",
            "--reserve",
            "--price-per-hour",
            "--fixed-state-bytes",
        ),
        at_least=(
            ("--context", 1),
            ("--memory-context", 1),
            ("--shared-prefix-tokens", 1),
            ("--batch", 1),
            ("--tp", 1),
        ),
    )
    if args.price_scope is not None and args.price_per_hour is None:
        raise ValueError(
            "--price-scope задаёт, к чему относится --price-per-hour; без цены он "
            "ни на что не влияет — добавьте --price-per-hour или уберите --price-scope"
        )
    tp = args.tp
    spec = _load_spec(args.config) if args.config is not None else None
    if tp > 1 and spec is None:
        raise ValueError(
            "для --tp нужен --config: деление KV между картами зависит от "
            "архитектуры (GQA — по головам KV, латентный KV MLA не делится — вывод "
            "из 2.3.2 и 6.2.2)"
        )
    kv_div = accounting.tp_kv_divisor(spec, tp) if spec is not None else 1
    state_total, state_div, state_notes = _serving_state(args, spec)
    context = args.context
    memory_context = args.memory_context or context
    if memory_context < context:
        raise ValueError(
            f"--memory-context ({memory_context}) короче --context ({context}): "
            "--context — длина для шага (обычно длина входа), --memory-context — "
            "длина к концу генерации для бюджета памяти (вход + выход − 1), она не "
            "может быть меньше; общий префикс задаёт --shared-prefix-tokens"
        )
    prefix = args.shared_prefix_tokens or 0
    if prefix > context:
        raise ValueError(
            f"--shared-prefix-tokens ({prefix}) длиннее --context ({context}): общий "
            "префикс — начало входа запроса и не может быть длиннее его"
        )
    if prefix and prefix == memory_context:
        raise ValueError(
            f"--shared-prefix-tokens ({prefix}) равен --memory-context: у запроса нет "
            "собственных токенов, а без них число запросов по памяти не определено; "
            "--memory-context — полная длина запроса к концу генерации"
        )
    kv_card = args.kv_per_token / kv_div
    state_card = state_total / state_div
    weights_card = args.weights / tp
    # окно внимания (sliding_window): запрос хранит и читает KV последних
    # min(длина, окно) токенов; из них общему префиксу принадлежат те, что не
    # собственные. Без окна — вся длина и весь префикс
    window = spec.window if spec is not None else None
    own_memory, shared_memory = _window_split(memory_context, prefix, window)
    own_step, shared_step = _window_split(context, prefix, window)

    dev = _device(args, "--peak-tflops, --bandwidth и --memory")
    if tp > 1 and dev is not None and dev.scope != SINGLE:
        raise ValueError(
            f"--tp {tp} с агрегатом {dev.id} ({dev.scope}, устройств в агрегате: "
            f"{dev.device_count}) посчитал бы карты дважды: ёмкость, полоса и пики агрегата уже "
            "суммированы по всем картам, а --tp ещё раз делит веса и KV. Выберите "
            "одиночное устройство из снимка (значения на одну карту) или задайте "
            "--memory, --bandwidth и --peak-tflops на одну карту без --device"
        )
    peak = _peak(args, dev)
    bw = _bandwidth(args, dev)
    _refuse_mixed_rates(dev, peak, bw)
    memory, skipped = _optional_memory(args, dev)
    rate_notes = _aggregate_note(dev, _labels(peak, bw), RATES_SUFFIX)
    if memory_context == context:
        lengths = (
            f"шаг и память — при одной длине context = {context} ток.: при длине "
            "входа память занижена, при длине к концу генерации шаг — граница только "
            "последнего шага; длину для памяти задаёт --memory-context"
        )
    else:
        lengths = (
            f"шаг — при context = {context} ток., память — при memory_context = "
            f"{memory_context} ток."
        )
    arch: dict[str, Any] = {}
    if spec is not None:
        arch = {"config": Path(args.config).name, "model_type": spec.model_type}
    parallel: dict[str, Any] = {"tp": tp} if tp > 1 else {}
    tp_notes: list[str] = []
    if tp > 1:
        assert spec is not None  # проверено выше: --tp > 1 требует --config
        split = (
            "не делится (латентный KV MLA; вывод из 2.3.2 и 6.2.2)"
            if kv_div == 1
            else f"делится на {kv_div}"
        )
        dup = (
            f", головы KV дублируются (копий каждой: {tp // spec.kv_heads})"
            if spec.family != "mla_moe" and tp > spec.kv_heads
            else ""
        )
        tp_notes.append(
            f"TP={tp}: веса, их чтение и FLOPs делятся на {tp} идеально, KV на токен "
            f"{split}{dup}; значения — на одну карту"
        )
        if state_div > 1:
            tp_notes.append(
                "фиксированное состояние S делится на TP: S/TP на карту, как в "
                "⌊(n·(80 GB − 2 GiB) − W)/S⌋ книги "
                "(references/source-book/chapter6.md:1164)"
            )
    state = {"fixed_state": state_total} if state_total else {}

    w_term = "M_w/TP" if tp > 1 else "M_w"
    kv_term = "kv_per_token/d_KV" if kv_div > 1 else "kv_per_token"
    s_term = (" + S/TP" if state_div > 1 else " + S") if state_total else ""
    window_input = {"window": window} if window else {}
    memory_inputs: dict[str, Any] = {
        **arch,
        "weights": args.weights,
        "kv_per_token": args.kv_per_token,
        "memory_context": memory_context,
        **({"shared_prefix": prefix} if prefix else {}),
        **window_input,
        "reserve": args.reserve,
        **state,
        **parallel,
    }
    if prefix:
        if window:
            own_term = "min(memory_context − shared_prefix, window)"
            shared_term = f"(min(memory_context, window) − {own_term})"
            own_text = f"{own_term} = {own_memory}"
        else:
            own_term = "(memory_context − shared_prefix)"
            shared_term = "shared_prefix"
            own_text = f"memory_context − shared_prefix = {own_memory}"
        memory_formula = (
            f"floor((C − {w_term} − reserve − {kv_term} × {shared_term}) / "
            f"({kv_term} × {own_term}{s_term}))"
        )
        if dev is not None and dev.scope != SINGLE:
            where = (
                "во всём агрегате"
                if memory is not None and memory.snapshot_label
                else "в пуле --memory"
            )
        else:
            where = "на карту"
        prefix_notes: tuple[str, ...] = (
            (
                f"общий префикс shared_prefix = {prefix} ток. хранится в пуле один раз "
                f"({format_number(round(kv_card * shared_memory))} B {where}), на "
                f"запрос — {own_text} собственных ток.; один раз "
                "хранятся заполненные блоки префикса, неполный хвостовой блок каждый "
                "запрос копирует при записи (8.3.2) — задавайте префикс кратным блоку KV"
            ),
        )
    else:
        length = "min(memory_context, window)" if window else "memory_context"
        memory_formula = (
            f"floor((C − {w_term} − reserve) / ({kv_term} × {length}{s_term}))"
        )
        prefix_notes = ()
    memory_window_notes = _window_notes(window, memory_context, "memory_context")
    if memory is None:
        capacity = _result(
            "max_concurrent_requests",
            None,
            "",
            "не вычисляется",
            memory_inputs,
            A_CH8_MEM,
            notes=(skipped,),
        )
    else:
        memory_notes = (
            *memory.notes,
            *_aggregate_note(
                dev,
                _labels(memory),
                "; веса и KV считаются размещёнными во всём агрегате",
            ),
            *(
                ("--reserve резервирует память на весь агрегат, а не на одну карту",)
                if dev is not None and dev.scope != SINGLE and memory.snapshot_label
                else ()
            ),
            "верхняя граница: активации, фрагментация и буферы сверх --reserve не учтены",
            lengths,
            *memory_window_notes,
            *prefix_notes,
            *tp_notes,
            *state_notes,
        )
        capacity = _result(
            "max_concurrent_requests",
            serving.max_concurrent_requests(
                memory.value,
                weights_card,
                kv_card * own_memory + state_card,
                args.reserve,
                shared_bytes=kv_card * shared_memory,
            ),
            "",
            memory_formula,
            memory.inputs | memory_inputs,
            A_CH8_SHARED if prefix else A_CH8_MEM,
            bound="upper",
            notes=memory_notes,
        )

    kv_key = "kv_request_per_device" if tp > 1 else "kv_request"
    kv_full = kv_card * (own_step + shared_step) + state_card
    # с общим префиксом нижняя граница читает его один раз на batch: как ядро
    # внимания читает префикс, книга не описывает, а меньше этого прочитать нельзя
    shared_read = kv_card * shared_step
    kv_request = kv_card * own_step + state_card

    def step_seconds(per_request: float, shared: float) -> float:
        return serving.tpot_lower_bound_seconds(
            args.batch,
            args.decode_flops / tp,
            args.weight_read / tp,
            per_request,
            peak.value,
            bw.value,
            shared_read_bytes=shared,
        )

    step = step_seconds(kv_request, shared_read)
    step_inputs = {
        **arch,
        "batch": args.batch,
        "decode_flops": args.decode_flops,
        "weight_read": args.weight_read,
        "context": context,
        **window_input,
        **(
            {"shared_prefix": prefix, "kv_shared_prefix": shared_read} if prefix else {}
        ),
        kv_key: kv_request,
        **state,
        **parallel,
        **peak.inputs,
        **bw.inputs,
    }
    f_term = "F_decode/(TP·Π)" if tp > 1 else "F_decode/Π"
    r_term = "R_W/TP" if tp > 1 else "R_W"
    kv_read = "R_KV/d_KV" if kv_div > 1 else "R_KV"
    kv_read = f"({kv_read}{s_term})" if state_total else kv_read
    step_formula = f"max(B·{f_term}, ({r_term} + B·{kv_read})/β)"
    if prefix:
        # префикс — один раз на batch, собственная часть и состояние — каждым запросом
        if window:
            own_tokens = "min(context − shared_prefix, window)"
            shared_tokens = f"(min(context, window) − {own_tokens})"
        else:
            own_tokens = "(context − shared_prefix)"
            shared_tokens = "shared_prefix"
        own_read = f"{kv_term} × {own_tokens}{s_term}"
        step_formula = (
            f"max(B·{f_term}, ({r_term} + {kv_term} × {shared_tokens} + "
            f"B·({own_read}))/β)"
        )
    step_notes = [
        *rate_notes,
        lengths,
        *_window_notes(window, context, "context"),
        *tp_notes,
    ]
    prefix_step_notes: list[str] = []
    if prefix:
        prefix_step_notes.append(
            "общий префикс читается один раз на batch — нижняя граница при любом ядре "
            "внимания; как ядро читает общий префикс, книга (8.3) не описывает. Если "
            "каждый запрос читает KV префикса сам, шаг не меньше "
            "tpot_without_prefix_dedup_seconds"
        )
    if tp > 1:
        step_notes.append(
            "AllReduce между картами не учтён (allreduce, ring): граница остаётся нижней"
        )
    if state_total:
        step_notes.append(
            "фиксированное состояние читается на каждом шаге (глава 2.3.6); запись "
            "не учтена, граница остаётся нижней"
        )
    throughput = serving.tokens_per_second(args.batch, step)
    out = [
        capacity,
        _result(
            "tpot_lower_bound_seconds",
            step,
            "s",
            step_formula,
            step_inputs,
            A_CH1_TIME,
            bound="lower",
            notes=(*step_notes, *prefix_step_notes),
        ),
        _result(
            "tokens_per_second_upper_bound",
            throughput,
            "tok/s",
            "B / T_step",
            step_inputs,
            A_CH1_TIME,
            bound="upper",
            notes=(*step_notes, *prefix_step_notes),
        ),
    ]
    if prefix:
        out.append(
            _result(
                "tpot_without_prefix_dedup_seconds",
                step_seconds(kv_full, 0),
                "s",
                f"max(B·{f_term}, ({r_term} + B·{kv_read})/β)",
                {
                    **{k: v for k, v in step_inputs.items() if k != "kv_shared_prefix"},
                    kv_key: kv_full,
                },
                A_CH1_TIME,
                notes=(
                    *step_notes,
                    (
                        "без дедупликации префикса: каждый запрос читает KV всех "
                        f"{'min(context, window)' if window else 'context'} ток., "
                        "включая общий префикс; нижняя граница шага, только "
                        "если ядро внимания читает префикс каждым запросом — это "
                        "допущение о ядре, книга (8.3) его не описывает"
                    ),
                ),
            )
        )
    if args.prefill_flops is not None:
        ttft = serving.ttft_lower_bound_seconds(
            args.prefill_flops / tp, weights_card, peak.value, bw.value
        )
        out.append(
            _result(
                "ttft_lower_bound_seconds",
                ttft,
                "s",
                "max(F_prefill/(TP·Π), M_w/(TP·β))"
                if tp > 1
                else "max(F_prefill/Π, M_w/β)",
                {
                    **arch,
                    "prefill_flops": args.prefill_flops,
                    "weights": args.weights,
                    **parallel,
                    **peak.inputs,
                    **bw.inputs,
                },
                A_CH1_REF,
                bound="lower",
                notes=(*rate_notes, *tp_notes),
            )
        )
    if args.price_per_hour is not None:
        # цена относится к агрегату, только если и скорость (пик, полоса) — его
        aggregate_rates = bool(_labels(peak, bw)) and dev is not None
        out.append(_serving_price(args, dev, aggregate_rates, tp, throughput, parallel))
    return out


def _training(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--total-tokens",),
        at_least=(("--tokens", 1), ("--dp", 1), ("--devices", 1)),
    )
    _check_fraction("--mfu", args.mfu, zero_allowed=False)
    spec = _load_spec(args.config)
    wrapper = _wrapper_notes(spec)
    _, _, total = flops.training_matrix_flops(spec, args.tokens)
    params = accounting.parameter_count(spec)
    active = accounting.parameter_count(spec, active=True)
    base = {"model_type": spec.model_type, "tokens": args.tokens}
    six_nd_notes = wrapper
    if active != params:
        six_nd_notes += (
            (
                f"MoE: N — {format_number(active)} активных параметров из "
                f"{format_number(params)}; все параметры "
                "определяют ёмкость (веса, состояние ZeRO), активные — лишь грубую "
                "оценку вычислений"
            ),
        )
    out = [
        _result(
            "training_flops_per_sequence",
            total,
            "FLOP",
            "3 × forward, словарная голова на всех токенах",
            base,
            A_CH3_TRAIN,
            notes=wrapper,
        ),
        _result(
            "six_nd_flops_per_sequence",
            flops.six_nd_flops(active, args.tokens),
            "FLOP",
            "6 × N_active × D",
            base | {"active_parameters": active},
            A_CH3_TRAIN,
            notes=six_nd_notes,
        ),
    ]
    if args.dp is not None:
        zero_note = (
            "байт на параметр: веса BF16 2, градиенты BF16 2, оптимизатор 12 "
            "(глава 10); глава 3.4.1 считает градиенты в FP32 — 18 байт"
        )
        for stage, formula in enumerate(ZERO_FORMULAS):
            state = training.sharded_state_bytes_per_device(params, args.dp, stage)
            out.append(
                _result(
                    f"zero{stage}_state_bytes_per_gpu",
                    state,
                    "B",
                    formula,
                    {"parameters": params, "dp": args.dp, "stage": stage},
                    A_CH10_ZERO,
                    notes=(zero_note, *wrapper),
                )
            )
    duration = {
        "--total-tokens": args.total_tokens,
        "--devices": args.devices,
        "--mfu": args.mfu,
    }
    if any(value is not None for value in duration.values()):
        missing = [flag for flag, value in duration.items() if value is None]
        if missing:
            raise ValueError(
                "для срока обучения нужны --total-tokens, --devices и --mfu; "
                f"не хватает: {', '.join(missing)}"
            )
        dev = _device(args, "--peak-tflops")
        if (
            dev is not None
            and dev.scope != SINGLE
            and args.peak_tflops is None
            and args.devices > 1
        ):
            raise ValueError(
                f"пик агрегата {dev.id} уже суммирован по всем его устройствам "
                f"({dev.device_count}), а --devices умножил бы его ещё раз: выберите одиночное "
                "устройство или задайте пик одной карты через --peak-tflops"
            )
        peak = _peak(args, dev)
        whole = total * args.total_tokens / args.tokens
        seconds = training.training_seconds(whole, args.devices, peak.value, args.mfu)
        inputs = {
            "total_flops": whole,
            "total_tokens": args.total_tokens,
            "devices": args.devices,
            "mfu": args.mfu,
            **peak.inputs,
        }
        note = f"FLOPs последовательности длиной {args.tokens} линейно масштабированы на все токены"
        out.append(
            _result(
                "training_seconds",
                seconds,
                "s",
                "F_total / (N · Π · MFU)",
                inputs,
                A_CH7_TIME,
                notes=(note, *_aggregate_note(dev, _labels(peak)), *wrapper),
            )
        )
    return out


def _checkpoint(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=(
            "--checkpoint-bytes",
            "--save-bandwidth",
            "--device-mtbf",
            "--common-job-mtbf",
            "--interval",
        ),
        non_negative=("--recovery",),
        at_least=(("--devices", 1), ("--stages", 1), ("--microbatches", 1)),
    )
    if (args.stages is None) != (args.microbatches is None):
        raise ValueError(
            "для утилизации конвейера нужны оба значения: --stages и --microbatches"
        )
    save = args.checkpoint_bytes / args.save_bandwidth
    rate = training.job_failure_rate(
        args.devices, args.device_mtbf, args.common_job_mtbf
    )
    inputs: dict[str, Any] = {
        "save_seconds": save,
        "failure_rate": rate,
        "recovery": args.recovery,
    }
    anchor = A_CH10_CKPT
    rate_notes: tuple[str, ...] = ()
    if args.common_job_mtbf is not None:
        inputs["common_job_mtbf"] = args.common_job_mtbf
        anchor = A_CH10_SPIKE
        rate_notes = (
            (
                "λ = N/MTBF_устройства + 1/MTBF_задания: общий для задания поток "
                "(откат из-за всплеска потерь) прибавляется один раз, а не на устройство"
            ),
        )
    first = training.checkpoint_optimal_interval(save, rate)
    interval = first if args.interval is None else args.interval
    loss_notes = (
        ()
        if args.interval is not None
        else ("τ — оптимум первого порядка; другой интервал задайте --interval",)
    )
    out = [
        _result(
            "first_order_optimal_interval_seconds",
            first,
            "s",
            "sqrt(2c/λ)",
            inputs,
            anchor,
            notes=rate_notes,
        ),
        _result(
            "poisson_optimal_interval_seconds",
            training.checkpoint_poisson_optimal_interval(save, rate),
            "s",
            "y − 1 + exp(−y − λc) = 0, τ = y/λ",
            inputs,
            anchor,
            notes=rate_notes,
        ),
        _result(
            "first_order_loss_at_interval",
            training.checkpoint_first_order_loss(interval, save, rate, args.recovery),
            "",
            "c/τ + λτ/2 + λr",
            inputs | {"interval": interval},
            anchor,
            notes=(*rate_notes, *loss_notes),
        ),
    ]
    if args.stages is not None:
        utilization = training.pipeline_utilization(args.stages, args.microbatches)
        out.append(
            _result(
                "pipeline_utilization",
                utilization,
                "",
                "m / (m + p − 1)",
                {"stages": args.stages, "microbatches": args.microbatches},
                A_CH10_PIPE,
                notes=("одинаковые стадии, передачи между стадиями не учтены",),
            )
        )
    return out


def _pipeline(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        non_negative=("--forward-seconds", "--backward-seconds"),
        at_least=(("--stages", 1), ("--microbatches", 1), ("--virtual-stages", 1)),
    )
    p, m, v = args.stages, args.microbatches, args.virtual_stages
    shape = {"stages": p, "microbatches": m, "virtual_stages": v}
    schedule = (
        "fill–drain и 1F1B"
        if v == 1
        else f"чередующийся 1F1B, блоков слоёв на карту: v = {v}"
    )
    ideal = (
        f"{schedule}; одинаковые стадии, передачи между стадиями и обновление "
        "параметров не учтены"
    )
    out = [
        _result(
            "pipeline_bubble_ratio",
            training.pipeline_bubble_ratio(p, m, v),
            "",
            "(p − 1) / (v · m)",
            shape,
            A_CH10_PIPE,
            notes=("доля пузыря относительно полезной работы, а не всего шага", ideal),
        ),
        _result(
            "pipeline_utilization",
            training.pipeline_utilization(p, m, v),
            "",
            "m / (m + (p − 1)/v)",
            shape,
            A_CH10_PIPE,
            bound="upper",
            notes=(ideal,),
        ),
    ]
    times = {
        "--forward-seconds": args.forward_seconds,
        "--backward-seconds": args.backward_seconds,
    }
    if any(value is not None for value in times.values()):
        missing = [flag for flag, value in times.items() if value is None]
        if missing:
            raise ValueError(
                "для времени шага нужны --forward-seconds и --backward-seconds; "
                f"не хватает: {', '.join(missing)}"
            )
        timed = shape | {
            "forward_seconds": args.forward_seconds,
            "backward_seconds": args.backward_seconds,
        }
        measured = (
            "в примере главы 10.3.2 формула даёт 330 ms, а событийная модель с "
            "передачами 1 ms и обновлением 1 ms — 337 ms (fill–drain), 347 ms (1F1B), "
            "298 ms (чередующийся, v = 2)"
        )
        out += [
            _result(
                "pipeline_step_seconds",
                training.pipeline_step_seconds(
                    p, m, args.forward_seconds, args.backward_seconds, v
                ),
                "s",
                "(m + (p − 1)/v)(t_f + t_b)",
                timed,
                A_CH10_PIPE,
                bound="lower",
                notes=(ideal, measured),
            ),
            _result(
                "pipeline_bubble_seconds",
                training.pipeline_bubble_seconds(
                    p, args.forward_seconds, args.backward_seconds, v
                ),
                "s",
                "(p − 1)(t_f + t_b) / v",
                timed,
                A_CH10_PIPE,
                bound="lower",
                notes=("простой каждой стадии за шаг", ideal),
            ),
        ]
    return out


def _training_state(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        non_negative=("--weight-bytes", "--grad-bytes", "--optimizer-bytes"),
        at_least=(("--dp", 1),),
    )
    if (args.config is None) == (args.params is None):
        raise ValueError(
            "укажите одно из двух: --config (число параметров по config.json) "
            "или --params (число параметров из карточки модели)"
        )
    notes: tuple[str, ...] = ()
    if args.config is not None:
        spec = _load_spec(args.config)
        try:
            params = accounting.parameter_count(spec)
        except UnsupportedArchitecture as error:
            raise ValueError(
                f"{error}; для training-state передайте --params"
            ) from None
        notes = _wrapper_notes(spec)
        if spec.experts:
            notes += ("MoE: состояние хранит всех экспертов, а не активных",)
    else:
        params = args.params
    parts = training.sharded_state_components(
        params,
        args.dp,
        args.stage,
        args.weight_bytes,
        args.grad_bytes,
        args.optimizer_bytes,
    )
    inputs = {
        "parameters": params,
        "dp": args.dp,
        "stage": args.stage,
        "weight_bytes_per_param": args.weight_bytes,
        "grad_bytes_per_param": args.grad_bytes,
        "optimizer_bytes_per_param": args.optimizer_bytes,
    }
    scope = (
        "постоянно размещённое состояние: активации, буферы AllGather и временные "
        "полные градиенты не входят; FSDP с полным шардированием соответствует stage 3"
    )
    rows = (
        (
            "weight_state_bytes_per_device",
            "P × b_w" + (" / d" if args.stage >= 3 else ""),
        ),
        (
            "gradient_state_bytes_per_device",
            "P × b_g" + (" / d" if args.stage >= 2 else ""),
        ),
        (
            "optimizer_state_bytes_per_device",
            "P × b_o" + (" / d" if args.stage >= 1 else ""),
        ),
    )
    out = [
        _result(name, value, "B", formula, inputs, A_CH10_ZERO, notes=notes)
        for (name, formula), value in zip(rows, parts)
    ]
    out.append(
        _result(
            "training_state_bytes_per_device",
            sum(parts),
            "B",
            "веса + градиенты + состояние оптимизатора на устройство",
            inputs,
            A_CH10_ZERO,
            notes=(scope, *notes),
        )
    )
    return out


def _rounds(values: Sequence[str] | None) -> list[tuple[float, int]]:
    rounds: list[tuple[float, int]] = []
    example = "0.0015:5"
    for text in values or ():
        seconds, sep, tokens = text.partition(":")
        try:
            if not sep:
                raise ValueError(text)
            parsed = (float(seconds), int(tokens))
        except ValueError:
            raise ValueError(
                f"--round ожидает секунды:токены, например {example}: {text!r}"
            ) from None
        if not math.isfinite(parsed[0]) or parsed[0] < 0:
            raise ValueError(
                f"--round ожидает конечное неотрицательное время раунда: {text!r}"
            )
        if parsed[1] < 1:
            raise ValueError(
                f"--round ожидает не меньше одного выходного токена за раунд: {text!r}"
            )
        rounds.append(parsed)
    return rounds


def _speculative(args: argparse.Namespace) -> list[Result]:
    _check_fraction("--acceptance", args.acceptance, zero_allowed=True)
    _check(args, positive=("--plain-step",), at_least=(("--draft", 0),))
    rounds = _rounds(args.rounds)
    plan = {
        "--acceptance": args.acceptance,
        "--draft": args.draft,
        "--plain-step": args.plain_step,
    }
    if not rounds and all(value is None for value in plan.values()):
        raise ValueError(
            "укажите --acceptance, --draft и --plain-step (ожидаемые токены за раунд "
            "и безубыточный раунд) или --round секунды:токены (среднее время на "
            "токен по измеренным раундам)"
        )
    out: list[Result] = []
    if any(value is not None for value in plan.values()):
        missing = [flag for flag, value in plan.items() if value is None]
        if missing:
            raise ValueError(
                "для E[N] и безубыточного раунда нужны --acceptance, --draft и "
                f"--plain-step; не хватает: {', '.join(missing)}"
            )
        tokens = speculative.expected_tokens_per_round(args.acceptance, args.draft)
        inputs = {"acceptance": args.acceptance, "draft": args.draft}
        out += [
            _result(
                "expected_tokens_per_round",
                tokens,
                "tok",
                "Σ a^i, i = 0..k",
                inputs,
                A_CH8_SPEC,
                notes=("позиции черновика принимаются независимо",),
            ),
            _result(
                "breakeven_round_seconds",
                speculative.breakeven_round_seconds(tokens, args.plain_step),
                "s",
                "E[N] × T_plain",
                inputs | {"plain_step": args.plain_step},
                A_CH8_SPEC,
                bound="upper",
                notes=("раунд длиннее этого времени медленнее обычного decode",),
            ),
        ]
    if rounds:
        seconds = [t for t, _ in rounds]
        counts = [n for _, n in rounds]
        out.append(
            _result(
                "mean_time_per_token_seconds",
                speculative.mean_time_per_token(seconds, counts),
                "s",
                "Σ T_r / Σ N_r",
                {
                    "rounds": ", ".join(f"{t:g}:{n}" for t, n in rounds),
                    "rounds_seconds": sum(seconds),
                    "total_tokens": sum(counts),
                },
                A_CH8_SPEC,
                notes=(
                    (
                        "среднее по токенам, а не по раундам: раунд с пятью токенами "
                        "весит впятеро больше раунда с одним"
                    ),
                ),
            )
        )
    return out


def _batch_threshold(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        non_negative=("--weight-read", "--kv-per-token", "--decode-flops"),
        at_least=(("--context", 1),),
    )
    raw = {
        "--weight-read": args.weight_read,
        "--kv-per-token": args.kv_per_token,
        "--decode-flops": args.decode_flops,
    }
    context = args.context
    notes: list[str] = []
    bound: str | None = None
    flops_refusal: str | None = None
    base: dict[str, Any] = {}
    if args.config is not None:
        given = [flag for flag, value in raw.items() if value is not None]
        if given:
            raise ValueError(
                "укажите либо --config, либо числа --weight-read, --kv-per-token и "
                f"--decode-flops; вместе с --config заданы: {', '.join(given)}"
            )
        spec = _load_spec(args.config)
        wb = units.dtype_bytes(args.weight_dtype)
        kb = units.dtype_bytes(args.kv_dtype)
        try:
            weight_read = float(accounting.decode_weight_read_bytes(spec, wb))
        except UnsupportedArchitecture as error:
            raise ValueError(
                f"{error}; передайте --weight-read, --kv-per-token и --decode-flops"
            ) from None
        kv = float(accounting.kv_bytes_per_token(spec, kb))
        tokens = min(context, spec.window) if spec.window else context
        if tokens != context:
            notes.append(f"окно внимания: читается KV последних токенов: {tokens}")
        try:
            flops_value: float | None = float(
                flops.forward_matrix_flops(spec, 1, context - 1)
            )
        except UnsupportedArchitecture as error:
            flops_value, flops_refusal = None, str(error)
        base = {
            "config": Path(args.config).name,
            "model_type": spec.model_type,
            "weight_dtype": args.weight_dtype,
            "kv_dtype": args.kv_dtype,
        }
        notes += _wrapper_notes(spec)
        if spec.experts:
            bound = "lower"
            notes.append(
                "MoE: R_W — эксперты одного токена; при батче читается объединение "
                "экспертов (model --batch), поэтому настоящий порог не меньше"
            )
        if wb < BF16_BYTES:
            bound = "lower"
            notes.append(QUANT_READ_NOTE)
    else:
        if args.weight_read is None or args.kv_per_token is None:
            raise ValueError(
                "укажите --config или --weight-read и --kv-per-token "
                "(для вычислительного порога ещё --decode-flops)"
            )
        weight_read, kv, flops_value = (
            args.weight_read,
            args.kv_per_token,
            args.decode_flops,
        )
        tokens = context
    inputs = base | {"weight_read": weight_read, "kv_per_token": kv, "context": context}
    kv_notes = list(notes)
    if kv * tokens > 0:
        threshold: int | None = roofline.batch_threshold(weight_read, kv, tokens)
    else:
        threshold = None
        kv_notes.append("без KV чтение контекста не растёт с батчем: порога нет")
    out = [
        _result(
            "kv_read_batch_threshold",
            threshold,
            "",
            "ceil(R_W / (kv_per_token × context))",
            inputs,
            A_CH2_MEM,
            bound=bound if threshold is not None else None,
            notes=(
                "батч, с которого чтение KV всех запросов не меньше чтения общих весов",
                *kv_notes,
            ),
        )
    ]
    rates = (args.device, args.peak_tflops, args.bandwidth)
    if all(value is None for value in rates):
        if args.config is None and args.decode_flops is not None:
            raise ValueError(
                "для вычислительного порога укажите --device или --peak-tflops и "
                "--bandwidth"
            )
        return out
    dev = _device(args, "--peak-tflops и --bandwidth")
    peak = _peak(args, dev)
    bw = _bandwidth(args, dev)
    _refuse_mixed_rates(dev, peak, bw)
    rate_notes = _aggregate_note(dev, _labels(peak, bw), RATES_SUFFIX)
    compute_inputs = inputs | {**peak.inputs, **bw.inputs}
    memory_note = (
        "ёмкость памяти не проверяется: KV всех запросов батча должен поместиться "
        "(serving)"
    )
    if flops_value is None:
        if flops_refusal is None:
            raise ValueError("для вычислительного порога нужен --decode-flops")
        out.append(
            _result(
                "compute_bound_batch_threshold",
                None,
                "",
                "не вычисляется",
                compute_inputs,
                A_CH1_TIME,
                notes=(flops_refusal, *notes),
            )
        )
        return out
    compute_inputs["decode_flops"] = flops_value
    kv_read = kv * tokens
    point = roofline.compute_bound_batch(
        flops_value, weight_read, kv_read, peak.value, bw.value
    )
    if point is None:
        value = None
        why = (
            "шаг decode не становится вычислительно ограниченным ни при каком батче: "
            f"на запрос F/Π = {flops_value / peak.value:.4g} s не больше "
            f"R_KV/β = {kv_read / bw.value:.4g} s"
        )
    else:
        value = max(1, math.ceil(point))
        why = f"B_* = {point:.4g}; минимальный целый батч — ceil(B_*)"
    generalisation = (
        "обобщение B_* = b_W·Π/(2β) книги (references/source-book/chapter1.md:305, "
        "без KV) на чтение KV каждого запроса; при R_KV = 0 совпадает с ним"
    )
    out.append(
        _result(
            "compute_bound_batch_threshold",
            value,
            "",
            "ceil((R_W/β) / (F/Π − R_KV/β)), R_KV = kv_per_token × context",
            compute_inputs,
            A_CH1_TIME,
            bound=bound if value is not None else None,
            notes=(why, generalisation, memory_note, *notes, *rate_notes),
        )
    )
    return out


def _allreduce(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--bandwidth",),
        non_negative=("--message", "--alpha"),
        at_least=(("--devices", 1),),
    )
    n, message, bw, alpha = args.devices, args.message, args.bandwidth, args.alpha
    inputs = {"devices": n, "message": message, "bandwidth": bw, "alpha": alpha}
    ring = collectives.ring_allreduce_seconds(n, message, bw, alpha)
    tree = collectives.tree_allreduce_seconds(n, message, bw, alpha)
    rounds = collectives.tree_allreduce_rounds(n)
    tree_note = (
        f"несегментированное биномиальное дерево, раундов: {rounds}; при n не степени "
        "двойки — ceil(log2 n), как tree_collective.py автора"
    )
    if ring > 0:
        ratio: float | None = tree / ring
        winner = (
            "дерево быстрее"
            if tree < ring
            else ("кольцо быстрее" if tree > ring else "время равно")
        )
        ratio_notes: tuple[str, ...] = (winner,)
    else:
        ratio, ratio_notes = None, ("одно устройство: коммуникации нет",)
    crossover = collectives.ring_tree_crossover_bytes(n, bw, alpha)
    return [
        _result(
            "ring_allreduce_seconds",
            ring,
            "s",
            "2(n−1)α + 2(n−1)M/(nB)",
            inputs,
            A_CH6_RING,
        ),
        _result(
            "tree_allreduce_seconds",
            tree,
            "s",
            "2·ceil(log2 n)·(α + M/B)",
            inputs,
            A_CH6_RING,
            notes=(tree_note,),
        ),
        _result(
            "tree_to_ring_time_ratio",
            ratio,
            "",
            "T_tree / T_ring",
            inputs,
            A_CH6_RING,
            notes=ratio_notes,
        ),
        _result(
            "ring_tree_crossover_bytes",
            crossover,
            "B",
            "αB(L − (n − 1)) / ((n − 1)/n − L), L = ceil(log2 n)",
            {"devices": n, "bandwidth": bw, "alpha": alpha},
            A_CH6_RING,
            notes=(
                "сообщение меньше этого объёма быстрее передаёт дерево, больше — кольцо"
                if crossover
                else "дерево не быстрее кольца ни при каком объёме",
            ),
        ),
    ]


def _ring(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--bandwidth",),
        non_negative=("--message", "--alpha"),
        at_least=(("--devices", 1),),
    )
    inputs = {
        "devices": args.devices,
        "message": args.message,
        "bandwidth": args.bandwidth,
        "alpha": args.alpha,
    }
    return [
        _result(
            "ring_allreduce_seconds",
            collectives.ring_allreduce_seconds(
                args.devices, args.message, args.bandwidth, args.alpha
            ),
            "s",
            "2(n−1)α + 2(n−1)M/(nB)",
            inputs,
            A_CH6_RING,
        ),
        _result(
            "bytes_sent_per_device",
            collectives.ring_bytes_sent_per_device(args.devices, args.message),
            "B",
            "2(n−1)M/n",
            inputs,
            A_CH6_RING,
        ),
    ]


def _cost(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--calls",),
        non_negative=(
            "--input-tokens",
            "--output-tokens",
            "--cached-tokens",
            "--cache-write-tokens",
            "--input-price",
            "--output-price",
            "--cache-read-price",
            "--cache-write-price",
        ),
    )
    _check_fraction("--success", args.success, zero_allowed=False)
    per_call = cost.call_cost(
        args.input_tokens,
        args.output_tokens,
        args.input_price,
        args.output_price,
        args.cached_tokens,
        args.cache_read_price,
        args.cache_write_tokens,
        args.cache_write_price,
    )
    tokens = {
        "input": args.input_tokens,
        "output": args.output_tokens,
        "cached": args.cached_tokens,
        "cache_write": args.cache_write_tokens,
    }
    prices = {
        "input_price": args.input_price,
        "output_price": args.output_price,
        "cache_read_price": args.cache_read_price,
        "cache_write_price": args.cache_write_price,
    }
    return [
        _result(
            "cost_per_call",
            per_call,
            "$",
            "(I_u·p_u + I_w·p_w + I_h·p_h + O·p_o) / 10^6",
            tokens | prices,
            A_CH11_CALL,
            notes=(
                "input — только некэшированный вход; хранение кэша и инструменты не входят",
            ),
        ),
        _result(
            "cost_per_accepted_task",
            cost.cost_per_accepted_task(per_call * args.calls, args.success),
            "$",
            "calls × C_call / p_success",
            {"cost_per_call": per_call, "calls": args.calls, "success": args.success},
            A_CH11_TASK,
        ),
    ]


def _edge(args: argparse.Namespace) -> list[Result]:
    _check(
        args, positive=("--up-mbps", "--down-mbps"), non_negative=("--rtt", "--compute")
    )
    up = units.megabits_per_second_to_bytes(args.up_mbps)
    down = units.megabits_per_second_to_bytes(args.down_mbps)
    upload = units.parse_bytes(args.upload)
    download = units.parse_bytes(args.download)
    inputs = {
        "upload": upload,
        "download": download,
        "up_mbps": args.up_mbps,
        "down_mbps": args.down_mbps,
        "rtt": args.rtt,
        "compute": args.compute,
    }
    seconds = edge.serial_seconds(upload, up, args.rtt, args.compute, download, down)
    return [
        _result(
            "serial_seconds",
            seconds,
            "s",
            "S_u/B_u + R + T_c + S_d/B_d",
            inputs,
            A_CH12_EDGE,
        )
    ]


def _units(args: argparse.Namespace) -> list[Result]:
    _check(args, non_negative=("--mbps",))
    if args.size is None and args.mbps is None and args.dtype is None:
        raise ValueError("укажите хотя бы одно: --size, --mbps или --dtype")
    if args.to and args.size is None:
        raise ValueError("--to переводит объём из --size; укажите --size")
    out: list[Result] = []
    if args.size is not None:
        size = units.parse_bytes(args.size)
        out.append(
            _result(
                "size_bytes",
                size,
                "B",
                "число × множитель единицы (GB = 10^9 B, GiB = 2^30 B)",
                {"size_text": args.size},
                A_CH1_UNITS,
            )
        )
        for unit in args.to or ():
            out.append(
                _result(
                    f"size_{unit}",
                    units.to_unit(size, unit),
                    unit,
                    f"size_bytes / {units.UNITS[unit]}",
                    {"size_bytes": size},
                    A_CH1_UNITS,
                )
            )
    if args.mbps is not None:
        out.append(
            _result(
                "link_bytes_per_second",
                units.megabits_per_second_to_bytes(args.mbps),
                "B/s",
                "Mbit/s × 10^6 / 8",
                {"mbps": args.mbps},
                A_CH1_UNITS,
            )
        )
    if args.dtype is not None:
        out.append(
            _result(
                "dtype_bytes",
                units.dtype_bytes(args.dtype),
                "B",
                "байт на один элемент",
                {"dtype": args.dtype},
                A_CH1_UNITS,
            )
        )
    return out


def _pairs(values: Sequence[str] | None, flag: str, example: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for text in values or ():
        name, sep, value = text.partition("=")
        if not sep or not name or not value:
            raise ValueError(
                f"{flag} ожидает имя=значение, например {example}: {text!r}"
            )
        pairs[name] = value
    return pairs


def _classes(values: Sequence[str] | None) -> dict[str, tuple[int, int]]:
    classes: dict[str, tuple[int, int]] = {}
    example = "long_input=8192:256"
    for name, value in _pairs(values, "--class", example).items():
        inputs, sep, outputs = value.partition(":")
        try:
            if not sep:
                raise ValueError(value)
            classes[name] = (int(inputs), int(outputs))
        except ValueError:
            raise ValueError(
                f"--class ожидает имя=входные:выходные токены, например {example}: "
                f"{name}={value!r}"
            ) from None
    return classes


def _rates(values: Sequence[str] | None) -> dict[str, float]:
    rates: dict[str, float] = {}
    for name, value in _pairs(values, "--rate", "long_input=2").items():
        try:
            rates[name] = float(value)
        except ValueError:
            raise ValueError(
                f"--rate ожидает интенсивность в запросах/с: {name}={value!r}"
            ) from None
    return rates


def _queueing(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--decode-capacity", "--prefill-capacity"),
        non_negative=("--arrival-rate", "--time-in-system"),
    )
    classes = _classes(args.classes)
    rates = _rates(args.rates)
    little = {
        "--arrival-rate": args.arrival_rate,
        "--time-in-system": args.time_in_system,
    }
    if not rates and not classes and all(v is None for v in little.values()):
        raise ValueError(
            "укажите --class и --rate (потребность по классам) "
            "или --arrival-rate и --time-in-system (закон Литтла)"
        )
    demand_anchor = QUEUE_ANCHORS.get(args.anchor, A_CH3_QUEUE)
    little_anchor = QUEUE_ANCHORS.get(args.anchor, A_CH11_LITTLE)
    out: list[Result] = []
    if rates or classes:
        if not rates or not classes:
            raise ValueError("потребность считается по --class и --rate вместе")
        tokens, steps = queueing.demand(rates, classes)
        inputs = {
            "classes": ", ".join(f"{k}={i}:{o}" for k, (i, o) in classes.items()),
            "rates": ", ".join(f"{k}={v:g}" for k, v in rates.items()),
        }
        out += [
            _result(
                "input_tokens_per_second",
                tokens,
                "tok/s",
                "Σ λ_c · I_c",
                inputs,
                demand_anchor,
            ),
            _result(
                "decode_steps_per_second",
                steps,
                "step/s",
                "Σ λ_c · (O_c − 1)",
                inputs,
                demand_anchor,
                notes=("первый выходной токен даёт prefill",),
            ),
        ]
        overload = "больше 1 — очередь растёт, сколько бы ни длилось окно"
        if args.capacity_upper_bound and (
            args.decode_capacity is None and args.prefill_capacity is None
        ):
            raise ValueError(
                "--capacity-upper-bound помечает загрузку, а без --decode-capacity "
                "или --prefill-capacity загрузка не считается"
            )
        utilization_bound = "lower" if args.capacity_upper_bound else None
        utilization_notes: tuple[str, ...] = (overload,)
        if args.capacity_upper_bound:
            utilization_notes += (
                (
                    "мощность — верхняя граница (например, "
                    "tokens_per_second_upper_bound из serving), поэтому загрузка — "
                    "нижняя граница: реальная не меньше"
                ),
            )
        for name, demand_key, demand, key, capacity in (
            (
                "decode_utilization",
                "decode_steps_per_second",
                steps,
                "decode_capacity",
                args.decode_capacity,
            ),
            (
                "prefill_utilization",
                "input_tokens_per_second",
                tokens,
                "prefill_capacity",
                args.prefill_capacity,
            ),
        ):
            if capacity is not None:
                out.append(
                    _result(
                        name,
                        queueing.utilization(demand, capacity),
                        "",
                        "потребность / мощность",
                        {demand_key: demand, key: capacity},
                        demand_anchor,
                        bound=utilization_bound,
                        notes=utilization_notes,
                    )
                )
    elif args.decode_capacity is not None or args.prefill_capacity is not None:
        raise ValueError("для загрузки нужны --class и --rate")
    elif args.capacity_upper_bound:
        raise ValueError(
            "--capacity-upper-bound относится к загрузке: нужны --class, --rate и мощность"
        )
    if any(v is not None for v in little.values()):
        missing = [flag for flag, value in little.items() if value is None]
        if missing:
            raise ValueError(
                "для закона Литтла нужны --arrival-rate и --time-in-system; "
                f"не хватает: {', '.join(missing)}"
            )
        out.append(
            _result(
                "in_system",
                queueing.littles_law_in_system(args.arrival_rate, args.time_in_system),
                "",
                "L = λ · W",
                {
                    "arrival_rate": args.arrival_rate,
                    "time_in_system": args.time_in_system,
                },
                little_anchor,
                notes=("среднее в устойчивом режиме",),
            )
        )
    return out


def _peak_label(peak: Mapping[str, Any]) -> str:
    fields = ("input_precision", "accumulator_precision", "execution_unit", "sparsity")
    unit = "TFLOP/s" if peak.get("operation_kind") == "floating_point" else "TOPS"
    label = "/".join(str(peak.get(field)) for field in fields)
    return f"{label}: {peak['tera_ops_per_second']} {unit}"


def _device_info(args: argparse.Namespace) -> list[Result]:
    dev = hardware.device(args.device, allow_aggregate=True)
    info = {
        "device": dev.id,
        "name": dev.name,
        "scope": dev.scope,
        "device_count": dev.device_count,
        "shared_with_cpu": dev.shared_with_cpu,
    }
    scope = _aggregate_note(
        dev,
        ["ёмкость памяти", "пропускная способность памяти", "пики"],
        "; на одно устройство снимок их не делит",
    )
    memory_notes = [
        *scope,
        "номинальная ёмкость по этикетке, а не доступный среде выполнения объём",
    ]
    if dev.shared_with_cpu:
        memory_notes.append("память общая с CPU и ОС: модели доступна только её часть")
    if dev.memory_bytes is None:
        memory_notes.append(f"ёмкость не опубликована в снимке {hardware.SNAPSHOT}")
    bandwidth_notes = [*scope]
    if dev.bandwidth is None:
        bandwidth_notes.append(
            f"пропускная способность не опубликована в снимке {hardware.SNAPSHOT}"
        )
    try:
        peak: float | None = hardware.peak_flops(dev)
        peak_notes = [*scope]
    except ValueError as error:
        peak, peak_notes = None, [*scope, str(error)]
    peak_notes += [
        _peak_label(p) for p in dev.peaks if p.get("tera_ops_per_second") is not None
    ]
    return [
        _result(
            "device_count",
            dev.device_count,
            "",
            "spec_scope и gpu_count/npu_count снимка",
            info,
            A_CH1_KEYS,
            notes=scope,
        ),
        _result(
            "memory_bytes",
            dev.memory_bytes,
            "B",
            "nominal_capacity × единица ёмкости",
            info,
            A_CH1_KEYS,
            notes=memory_notes,
        ),
        _result(
            "bandwidth",
            dev.bandwidth,
            "B/s",
            "bandwidth_bytes_per_second",
            info,
            A_CH1_KEYS,
            notes=bandwidth_notes,
        ),
        _result(
            "peak_flops",
            peak,
            "FLOP/s",
            "BF16 / накопление FP32 / tensor / dense; все пики снимка — в примечаниях (вход/накопление/блок/sparsity)",
            info,
            A_CH1_KEYS,
            notes=peak_notes,
        ),
    ]


# --- разбор аргументов ---------------------------------------------------------


class _UsageError(Exception):
    """Ошибка разбора аргументов: сообщение уже переведено."""


_USAGE_MESSAGES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"the following arguments are required: (?P<a>.+)"),
        "не заданы обязательные аргументы: {a}",
    ),
    (re.compile(r"unrecognized arguments: (?P<a>.+)"), "неизвестные аргументы: {a}"),
    (
        re.compile(r"ambiguous option: (?P<a>\S+) could match (?P<b>.+)"),
        "неоднозначный флаг {a}: подходят {b}",
    ),
)
_ARGUMENT_MESSAGES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"invalid choice: (?P<a>.+?) \(choose from (?P<b>.+)\)"),
        "недопустимое значение {a}; допустимы: {b}",
    ),
    (
        re.compile(r"invalid \w+ value: (?P<a>.+)"),
        "ожидается число, получено {a}",
    ),
    (re.compile(r"expected one argument"), "нужно значение"),
)


def _translate(message: str) -> str:
    """Сообщения argparse — на русском; неизвестные передаются как есть."""
    argument = re.fullmatch(
        r"argument (?P<name>[^:]+): (?P<rest>.+)", message, re.DOTALL
    )
    if argument:
        rest = argument["rest"]
        for pattern, template in _ARGUMENT_MESSAGES:
            match = pattern.fullmatch(rest)
            if match:
                rest = template.format(**match.groupdict())
                break
        return f"аргумент {argument['name']}: {rest}"
    for pattern, template in _USAGE_MESSAGES:
        match = pattern.fullmatch(message)
        if match:
            return template.format(**match.groupdict())
    return f"ошибка аргументов: {message}"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(_translate(message))


def _count(text: str) -> int:
    """Целое число, записанное как угодно: 397e9, 397000000000."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ожидается число: {text!r}") from None
    if not math.isfinite(value) or not value.is_integer() or value < 1:
        raise argparse.ArgumentTypeError(f"ожидается целое число не меньше 1: {text!r}")
    return int(value)


def _parser() -> argparse.ArgumentParser:
    fmt = _Parser(add_help=False)
    fmt.add_argument(
        "--format",
        choices=("json", "md"),
        default=argparse.SUPPRESS,
        help="вывод: md (по умолчанию) или json",
    )
    parser = _Parser(
        prog="calc.py",
        description="Калькуляторы по книге «AI Infra in Depth». Каждый результат — "
        "с формулой, входными данными и якорем на книгу.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "md"),
        default="md",
        help="вывод: md (по умолчанию) или json",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="команда")

    def command(
        name: str,
        handler: Callable[[argparse.Namespace], list[Result]],
        help_text: str,
    ) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[fmt], help=help_text, description=help_text)
        p.set_defaults(handler=handler)
        return p

    def device_options(p: argparse.ArgumentParser, bandwidth: bool = True) -> None:
        p.add_argument("--device", help=f"id устройства из снимка {hardware.SNAPSHOT}")
        p.add_argument(
            "--allow-aggregate",
            action="store_true",
            help="разрешить стойку, сервер или суперчип: значения суммированы по устройствам",
        )
        p.add_argument("--precision", default="BF16", help="точность входа пика (BF16)")
        p.add_argument(
            "--accumulator",
            default="FP32",
            help="точность накопления пика (FP32, как в книге)",
        )
        p.add_argument("--sparsity", default="dense", choices=("dense", "structured"))
        p.add_argument(
            "--peak-tflops", type=float, help="пик явно, TFLOP/s (вместо снимка)"
        )
        if bandwidth:
            p.add_argument(
                "--bandwidth",
                type=float,
                help="пропускная способность памяти явно, байт/с",
            )

    p = command(
        "model", _model, "параметры, память весов, KV и FLOPs модели по config.json"
    )
    p.add_argument("--config", required=True, help="config.json в формате Hugging Face")
    p.add_argument("--context", type=int, default=0, help="длина контекста, токенов")
    p.add_argument("--weight-dtype", default="bf16")
    p.add_argument("--kv-dtype", default="bf16")
    p.add_argument(
        "--params",
        type=_count,
        help="число параметров из карточки модели, если калькулятор его не считает (гибридные модели)",
    )
    p.add_argument(
        "--quant-overhead-bytes",
        type=float,
        help="байт scale и частей в высокой точности сверх параметров × байты (квантизация)",
    )
    p.add_argument(
        "--tp",
        type=int,
        help="число карт тензорного параллелизма: веса, KV и состояние на карту",
    )
    p.add_argument(
        "--batch",
        type=int,
        help="запросов в шаге decode (MoE): объединение экспертов и чтение весов",
    )
    p.add_argument(
        "--experts-per-layer",
        type=float,
        help="различных экспертов на слой в батче вместо равномерной оценки "
        "(измерение или худший случай min(B·k, E))",
    )

    p = command("roofline", _roofline, "нижняя граница времени шага: max(F/Π, R/β)")
    device_options(p)
    p.add_argument(
        "--memory-only",
        action="store_true",
        help="только граница по чтению R/β, без пика (устройства без пика в снимке, "
        "например Apple); вычислительная граница не оценивается",
    )
    p.add_argument("--flops", type=float, required=True, help="FLOPs шага")
    p.add_argument(
        "--bytes", type=float, required=True, help="байт, прочитанных за шаг"
    )

    p = command(
        "serving", _serving, "память, TTFT, TPOT, пропускная способность и цена токена"
    )
    device_options(p)
    p.add_argument("--weights", type=float, required=True, help="байт весов в памяти")
    p.add_argument(
        "--weight-read",
        type=float,
        required=True,
        help="байт весов, читаемых за шаг decode",
    )
    p.add_argument(
        "--decode-flops", type=float, required=True, help="FLOPs шага decode на запрос"
    )
    p.add_argument(
        "--prefill-flops", type=float, help="FLOPs prefill одного запроса (для TTFT)"
    )
    p.add_argument(
        "--kv-per-token", type=float, required=True, help="байт KV на токен всей модели"
    )
    p.add_argument(
        "--context",
        type=int,
        required=True,
        help="длина контекста для шага, токенов: при длине входа — нижняя граница "
        "каждого шага decode",
    )
    p.add_argument(
        "--memory-context",
        type=int,
        help="длина контекста для бюджета памяти, токенов: к концу генерации (вход + "
        "выход − 1, до блока KV); по умолчанию равна --context",
    )
    p.add_argument(
        "--shared-prefix-tokens",
        type=int,
        help="общий префикс запросов, токенов (начало входа, в блоках KV): его KV "
        "хранится в пуле один раз, на запрос — --memory-context минус префикс; шаг "
        "по-прежнему при --context",
    )
    p.add_argument("--batch", type=int, default=1, help="запросов в шаге decode")
    p.add_argument(
        "--memory",
        type=float,
        help="байт памяти одного устройства; без него и без --device ёмкость не "
        "оценивается",
    )
    p.add_argument(
        "--reserve", type=float, default=0, help="байт резерва среды выполнения"
    )
    p.add_argument(
        "--fixed-state-bytes",
        type=float,
        help="байт фиксированного состояния на запрос всей модели (линейное внимание "
        "и т. п.); для гибридной модели из --config берётся автоматически",
    )
    p.add_argument(
        "--config",
        help="config.json модели: архитектура для --tp, фиксированного состояния и "
        "окна внимания (sliding_window ограничивает KV запроса)",
    )
    p.add_argument(
        "--tp",
        type=int,
        default=1,
        help="карт тензорного параллелизма; --weights, --weight-read, FLOPs и KV "
        "задаются для всей модели и делятся по картам (нужен --config)",
    )
    p.add_argument(
        "--price-per-hour",
        type=float,
        help="$ за час выбранной записи устройства: одной карты или, с "
        "--price-scope aggregate, всей записи агрегата",
    )
    p.add_argument(
        "--price-scope",
        choices=("card", "aggregate"),
        help="к чему относится --price-per-hour: card — за карту (по умолчанию; у "
        "агрегата из снимка умножается на число устройств), aggregate — за всю "
        "запись агрегата (только агрегат с пиком и полосой из снимка); у такого "
        "агрегата обязателен, нужен --price-per-hour",
    )

    p = command("training", _training, "FLOPs обучения, состояние ZeRO и срок обучения")
    device_options(p, bandwidth=False)
    p.add_argument("--config", required=True, help="config.json в формате Hugging Face")
    p.add_argument(
        "--tokens", type=int, required=True, help="длина обучающей последовательности"
    )
    p.add_argument("--dp", type=int, help="число GPU в группе шардирования ZeRO")
    p.add_argument("--total-tokens", type=float, help="всего токенов обучения")
    p.add_argument("--devices", type=int, help="число ускорителей")
    p.add_argument("--mfu", type=float, help="доля пика, (0, 1]")

    p = command("checkpoint", _checkpoint, "интервал checkpoint и утилизация конвейера")
    p.add_argument("--checkpoint-bytes", type=float, required=True)
    p.add_argument("--save-bandwidth", type=float, required=True, help="байт/с")
    p.add_argument("--devices", type=int, required=True)
    p.add_argument(
        "--device-mtbf",
        type=float,
        required=True,
        help="секунд между сбоями одного устройства",
    )
    p.add_argument("--recovery", type=float, default=0, help="секунд на восстановление")
    p.add_argument(
        "--interval",
        type=float,
        help="интервал для доли потерь, секунд (по умолчанию — оптимум)",
    )
    p.add_argument(
        "--common-job-mtbf",
        type=float,
        help="секунд между событиями, прерывающими всё задание сразу (например, "
        "откат из-за всплеска потерь); прибавляется к частоте один раз",
    )
    p.add_argument("--stages", type=int, help="стадий конвейера")
    p.add_argument("--microbatches", type=int, help="микропакетов на шаг")

    p = command(
        "pipeline",
        _pipeline,
        "пузырь и утилизация конвейера, время шага при одинаковых стадиях",
    )
    p.add_argument("--stages", type=int, required=True, help="стадий конвейера p")
    p.add_argument(
        "--microbatches", type=int, required=True, help="micro-batch на шаг m"
    )
    p.add_argument(
        "--virtual-stages",
        type=int,
        default=1,
        help="блоков слоёв на карту v для чередующегося 1F1B (1 — fill–drain и 1F1B)",
    )
    p.add_argument(
        "--forward-seconds", type=float, help="секунд прямого прохода одной стадии"
    )
    p.add_argument(
        "--backward-seconds", type=float, help="секунд обратного прохода одной стадии"
    )

    p = command(
        "training-state",
        _training_state,
        "веса, градиенты и состояние оптимизатора на устройство при ZeRO/FSDP",
    )
    p.add_argument("--config", help="config.json в формате Hugging Face")
    p.add_argument(
        "--params",
        type=_count,
        help="число параметров из карточки модели (вместо --config)",
    )
    p.add_argument(
        "--dp", type=int, required=True, help="число GPU в группе шардирования d"
    )
    p.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=(0, 1, 2, 3),
        help="stage ZeRO: 0 — обычный DP, 1 — оптимизатор, 2 — и градиенты, 3 — и веса",
    )
    p.add_argument(
        "--weight-bytes", type=float, default=2, help="байт весов на параметр (BF16: 2)"
    )
    p.add_argument(
        "--grad-bytes",
        type=float,
        default=2,
        help="байт градиентов на параметр (BF16: 2; глава 3.4.1 — FP32: 4)",
    )
    p.add_argument(
        "--optimizer-bytes",
        type=float,
        default=12,
        help="байт оптимизатора на параметр (основные веса FP32 и два момента Adam: 12)",
    )

    p = command(
        "speculative",
        _speculative,
        "спекулятивное декодирование: E[N], безубыточный раунд, время на токен",
    )
    p.add_argument("--acceptance", type=float, help="доля принятия позиции, [0, 1]")
    p.add_argument("--draft", type=int, help="токенов черновика за раунд")
    p.add_argument("--plain-step", type=float, help="секунд на обычный шаг decode")
    p.add_argument(
        "--round",
        dest="rounds",
        action="append",
        help="измеренный раунд секунды:выходные токены, например 0.0015:5 (можно "
        "несколько раз): среднее время на токен",
    )

    p = command("ring", _ring, "кольцевой AllReduce")
    p.add_argument("--devices", type=int, required=True)
    p.add_argument("--message", type=float, required=True, help="байт")
    p.add_argument(
        "--bandwidth", type=float, required=True, help="байт/с в одном направлении"
    )
    p.add_argument("--alpha", type=float, required=True, help="секунд на раунд")

    p = command("allreduce", _allreduce, "кольцевой и древовидный AllReduce: сравнение")
    p.add_argument("--devices", type=int, required=True, help="участников n")
    p.add_argument("--message", type=float, required=True, help="байт на участника M")
    p.add_argument(
        "--bandwidth", type=float, required=True, help="байт/с в одном направлении"
    )
    p.add_argument("--alpha", type=float, required=True, help="секунд на раунд")

    p = command(
        "batch-threshold",
        _batch_threshold,
        "батч, с которого чтение KV догоняет чтение весов и decode упирается в вычисления",
    )
    device_options(p)
    p.add_argument("--config", help="config.json в формате Hugging Face")
    p.add_argument("--weight-dtype", default="bf16", help="тип весов при --config")
    p.add_argument("--kv-dtype", default="bf16", help="тип KV при --config")
    p.add_argument(
        "--weight-read", type=float, help="байт общих весов, читаемых за шаг decode"
    )
    p.add_argument("--kv-per-token", type=float, help="байт KV на токен")
    p.add_argument(
        "--decode-flops",
        type=float,
        help="FLOPs шага decode на запрос (для вычислительного порога)",
    )
    p.add_argument(
        "--context", type=int, required=True, help="длина контекста, токенов"
    )

    p = command("cost", _cost, "стоимость вызова и принятой задачи")
    for name in (
        "input-tokens",
        "output-tokens",
        "cached-tokens",
        "cache-write-tokens",
    ):
        p.add_argument(
            f"--{name}", type=float, default=0, help="токенов (можно среднее)"
        )
    for name in (
        "input-price",
        "output-price",
        "cache-read-price",
        "cache-write-price",
    ):
        p.add_argument(f"--{name}", type=float, default=0, help="$ за миллион токенов")
    p.add_argument("--calls", type=float, default=1, help="вызовов на попытку задачи")
    p.add_argument(
        "--success", type=float, default=1.0, help="вероятность приёмки, (0, 1]"
    )

    p = command("edge", _edge, "полное время взаимодействия устройства и облака")
    p.add_argument("--upload", required=True, help="объём с единицей, например '30 MB'")
    p.add_argument(
        "--download", required=True, help="объём с единицей, например '5 MB'"
    )
    p.add_argument("--up-mbps", type=float, required=True)
    p.add_argument("--down-mbps", type=float, required=True)
    p.add_argument("--rtt", type=float, required=True, help="секунд")
    p.add_argument("--compute", type=float, required=True, help="секунд работы модели")

    p = command("units", _units, "перевод объёмов, скоростей канала и типов данных")
    p.add_argument("--size", help="объём с явной единицей, например '141.11 GB'")
    p.add_argument(
        "--to",
        action="append",
        choices=tuple(units.UNITS),
        help="в какую единицу перевести --size (можно несколько раз)",
    )
    p.add_argument("--mbps", type=float, help="скорость канала, Mbit/s")
    p.add_argument("--dtype", help="тип данных: bf16, fp8, int4, …")

    p = command(
        "queueing", _queueing, "потребность по классам запросов, загрузка, закон Литтла"
    )
    p.add_argument(
        "--class",
        dest="classes",
        action="append",
        help="класс запросов имя=входные:выходные токены, например long_input=8192:256",
    )
    p.add_argument(
        "--rate",
        dest="rates",
        action="append",
        help="интенсивность класса имя=запросов/с, например long_input=2",
    )
    p.add_argument("--decode-capacity", type=float, help="шагов decode в секунду")
    p.add_argument("--prefill-capacity", type=float, help="входных токенов в секунду")
    p.add_argument("--arrival-rate", type=float, help="поступлений в секунду")
    p.add_argument("--time-in-system", type=float, help="секунд в системе")
    p.add_argument(
        "--capacity-upper-bound",
        action="store_true",
        help="мощность — верхняя граница (из serving): загрузка помечается как "
        "нижняя граница",
    )
    p.add_argument(
        "--anchor",
        choices=tuple(QUEUE_ANCHORS),
        help="глава-якорь всех результатов: ch03 — нагрузка, ch08 — поток запросов "
        "инференса, ch11 — среды агентов; по умолчанию потребность — ch03, закон "
        "Литтла — ch11",
    )

    p = command(
        "device",
        _device_info,
        "запись снимка: память, пропускная способность, пики, охват",
    )
    p.add_argument("--device", required=True)
    return parser


def _error_text(error: Exception) -> str:
    # str(KeyError) заключает сообщение в кавычки
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def _requested_json(argv: Sequence[str]) -> bool:
    """Формат вывода для ошибки разбора, когда argparse ещё не вернул аргументы."""
    for index, item in enumerate(argv):
        if item == "--format=json":
            return True
        if item == "--format" and index + 1 < len(argv) and argv[index + 1] == "json":
            return True
    return False


def _print_error(message: str, as_json: bool, command: str | None) -> None:
    if as_json:
        print(json.dumps({"command": command, "error": message}, ensure_ascii=False))
    else:
        print(f"Ошибка: {message}")


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    try:
        args = parser.parse_args(raw)
    except _UsageError as error:
        hint = "" if _requested_json(raw) else ". Справка: calc.py <команда> --help"
        _print_error(f"{error}{hint}", _requested_json(raw), None)
        return 2
    except SystemExit as stop:
        # --help: argparse уже напечатал справку
        return stop.code if isinstance(stop.code, int) else 0
    as_json = args.format == "json"
    try:
        results = args.handler(args)
    except (ValueError, KeyError) as error:
        _print_error(_error_text(error), as_json, args.command)
        return 2
    snapshot = hardware.SNAPSHOT if getattr(args, "device", None) else None
    if as_json:
        payload = {
            "command": args.command,
            "hardware_snapshot": snapshot,
            "results": [result.to_dict() for result in results],
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        if snapshot:
            print(f"Снимок железа: {snapshot}\n")
        print("\n\n".join(result.to_markdown() for result in results))
    return 0
