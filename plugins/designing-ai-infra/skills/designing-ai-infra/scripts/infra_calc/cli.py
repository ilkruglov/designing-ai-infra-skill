"""CLI калькуляторов. Вывод: JSON или Markdown с формулой, входами и якорем.

Каждый результат несёт формулу, входные данные, тип границы и якорь на
заголовок книги. Если число взято из снимка железа, вывод называет снимок;
стойка или сервер из снимка берутся только с флагом --allow-aggregate.
Ошибка входа завершает команду с кодом 2 и сообщением на русском.

collectives.all_to_all_phase и collectives.tp_step_seconds вызываются из
Python: матрицу назначений неудобно передавать аргументами.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import (
    accounting,
    collectives,
    cost,
    edge,
    flops,
    hardware,
    roofline,
    serving,
    speculative,
    training,
    units,
)
from .checks import require_int_at_least, require_non_negative, require_positive
from .model import ModelSpec, UnsupportedArchitecture, load_config, parse_spec
from .result import Result

BOOK = "references/source-book"
A_CH1_KEYS = f"{BOOK}/chapter1.md:152"
A_CH1_TIME = f"{BOOK}/chapter1.md:263"
A_CH1_REF = f"{BOOK}/chapter1.md:450"
A_CH2_MEM = f"{BOOK}/chapter2.md:236"
A_CH2_HYBRID = f"{BOOK}/chapter2.md:418"
A_CH3_PRICE = f"{BOOK}/chapter3.md:209"
A_CH3_STATE = f"{BOOK}/chapter3.md:384"
A_CH3_TRAIN = f"{BOOK}/chapter3.md:442"
A_CH6_RING = f"{BOOK}/chapter6.md:473"
A_CH7_TIME = f"{BOOK}/chapter7.md:938"
A_CH8_MEM = f"{BOOK}/chapter8.md:52"
A_CH8_SPEC = f"{BOOK}/chapter8.md:536"
A_CH10_ZERO = f"{BOOK}/chapter10.md:151"
A_CH10_PIPE = f"{BOOK}/chapter10.md:322"
A_CH10_CKPT = f"{BOOK}/chapter10.md:553"
A_CH11_CALL = f"{BOOK}/chapter11.md:488"
A_CH11_TASK = f"{BOOK}/chapter11.md:515"
A_CH12_EDGE = f"{BOOK}/chapter12.md:11"

SINGLE = "single_device"
ZERO_FORMULAS = (
    "P × (2 + 2 + 12)",
    "P × (2 + 2 + 12/N)",
    "P × (2 + (2 + 12)/N)",
    "P × (2 + 2 + 12) / N",
)
USER = "задано пользователем"


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


def _aggregate_note(dev: hardware.Device, what: str) -> str:
    return (
        f"{dev.id} — агрегат {dev.scope} из {dev.device_count} устройств: "
        f"{what} суммированы по {dev.device_count} устройствам, "
        "а не значения одной карты"
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


@dataclass(frozen=True)
class _Rates:
    peak: float | None
    bandwidth: float | None
    inputs: dict[str, Any]
    notes: tuple[str, ...]
    device: hardware.Device | None


def _rates(
    args: argparse.Namespace, per_device: str, need_bandwidth: bool = True
) -> _Rates:
    """Пик и пропускная способность: явные аргументы важнее снимка."""
    dev = _device(args, per_device)
    peak_given = args.peak_tflops is not None
    bandwidth_given = need_bandwidth and args.bandwidth is not None
    if dev is None and not (peak_given and (bandwidth_given or not need_bandwidth)):
        needed = "--peak-tflops и --bandwidth" if need_bandwidth else "--peak-tflops"
        raise ValueError(f"укажите --device или {needed}")
    inputs: dict[str, Any] = {}
    notes: list[str] = []
    if dev is not None:
        inputs["device"] = dev.id
    if peak_given:
        require_positive("--peak-tflops", args.peak_tflops)
        peak = args.peak_tflops * 1e12
        inputs["peak_source"] = USER
    else:
        peak = _snapshot_peak(dev, args)
        inputs |= {
            "precision": args.precision,
            "accumulator": args.accumulator,
            "sparsity": args.sparsity,
        }
    inputs["peak"] = peak
    bandwidth = None
    if bandwidth_given:
        require_positive("--bandwidth", args.bandwidth)
        bandwidth = args.bandwidth
        inputs["bandwidth_source"] = USER
    elif need_bandwidth:
        bandwidth = hardware.bandwidth(dev)
    if need_bandwidth:
        inputs["bandwidth"] = bandwidth
    from_snapshot = not peak_given or (need_bandwidth and not bandwidth_given)
    if dev is not None and dev.scope != SINGLE and from_snapshot:
        notes.append(
            _aggregate_note(dev, "пик и пропускная способность")
            + "; граница предполагает идеальное деление работы между ними"
        )
    return _Rates(peak, bandwidth, inputs, tuple(notes), dev)


# --- команды -------------------------------------------------------------------


def _model(args: argparse.Namespace) -> list[Result]:
    spec = _load_spec(args.config)
    wb = units.dtype_bytes(args.weight_dtype)
    kb = units.dtype_bytes(args.kv_dtype)
    wrapper = _wrapper_notes(spec)
    base: dict[str, Any] = {
        "config": Path(args.config).name,
        "model_type": spec.model_type,
    }
    if spec.wrapper_model_type:
        base["wrapper_model_type"] = spec.wrapper_model_type
    out: list[Result] = []
    try:
        params: int | None = accounting.parameter_count(spec)
        refusal = ""
    except UnsupportedArchitecture as error:
        params, refusal = None, str(error)

    if params is not None:
        if args.params is not None:
            raise ValueError(
                "--params нужен только для архитектур, где калькулятор не считает "
                f"параметры; для {spec.model_type} по config.json получено {params}"
            )
        weights = {**base, "weight_dtype": args.weight_dtype}
        read_notes = wrapper
        if spec.experts:
            read_notes += (
                "эксперты одного токена; при батче читается объединение экспертов",
            )
        out += [
            Result(
                "parameters",
                params,
                "",
                "Σ матриц + нормализации + эмбеддинги (без MTP)",
                base,
                A_CH3_STATE,
                notes=wrapper,
            ),
            Result(
                "active_parameters",
                accounting.parameter_count(spec, active=True),
                "",
                "эксперты одного токена вместо всех",
                base,
                A_CH2_MEM,
                notes=wrapper,
            ),
            Result(
                "weight_bytes",
                accounting.weight_bytes(spec, wb),
                "B",
                "parameters × bytes_per_param",
                weights,
                A_CH1_REF,
                notes=wrapper,
            ),
            Result(
                "decode_weight_read_bytes",
                accounting.decode_weight_read_bytes(spec, wb),
                "B",
                "(active_parameters − таблица эмбеддингов без общих весов) × bytes_per_param",
                weights,
                A_CH2_MEM,
                notes=read_notes,
            ),
        ]
    elif args.params is not None:
        given = {**base, "params": args.params}
        out += [
            Result(
                "parameters",
                args.params,
                "",
                f"{USER} (--params)",
                given,
                A_CH3_STATE,
                notes=(refusal, *wrapper),
            ),
            Result(
                "weight_bytes",
                int(args.params * wb),
                "B",
                f"parameters ({USER}) × bytes_per_param",
                given | {"weight_dtype": args.weight_dtype},
                A_CH1_REF,
                notes=wrapper,
            ),
        ]
    else:
        out.append(
            Result(
                "parameters",
                None,
                "",
                "не вычисляется",
                base,
                A_CH3_STATE,
                notes=(refusal, *wrapper),
            )
        )

    kv_formula = {
        "mla_moe": "layers × (kv_lora_rank + qk_rope_dim) × bytes",
        "hybrid_linear": "2 × full_attention_layers × kv_heads × head_dim × bytes",
    }.get(spec.family, "2 × layers × kv_heads × head_dim × bytes")
    kv_inputs = {**base, "kv_dtype": args.kv_dtype}
    out.append(
        Result(
            "kv_bytes_per_token",
            accounting.kv_bytes_per_token(spec, kb),
            "B",
            kv_formula,
            kv_inputs,
            A_CH8_MEM,
        )
    )
    if args.context:
        resident = accounting.kv_resident_bytes(spec, args.context, kb)
        window = f"min(context, {spec.window})" if spec.window else "context"
        out.append(
            Result(
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
            Result(
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
    return out


def _forward_flops(spec: ModelSpec, context: int, base: dict[str, Any]) -> list[Result]:
    try:
        prefill = flops.forward_matrix_flops(spec, context)
        decode = flops.forward_matrix_flops(spec, 1, context - 1)
    except UnsupportedArchitecture as error:
        return [
            Result(
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
        Result(
            "prefill_flops",
            prefill,
            "FLOP",
            "2·W_active·n + 2·V·h + 4·L·H·d·n(n+1)/2",
            base | {"tokens": context},
            A_CH1_REF,
        ),
        Result(
            "decode_step_flops",
            decode,
            "FLOP",
            "2·W_active + 2·V·h + 4·L·H·d·context",
            base | {"context": context},
            A_CH1_REF,
        ),
    ]


def _roofline(args: argparse.Namespace) -> list[Result]:
    rates = _rates(args, "--peak-tflops и --bandwidth")
    peak, bw = rates.peak, rates.bandwidth
    inputs = {"flops": args.flops, "bytes": args.bytes, **rates.inputs}
    out = [
        Result(
            "step_lower_bound_seconds",
            roofline.lower_bound_seconds(args.flops, args.bytes, peak, bw),
            "s",
            "max(F/Π, R/β)",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=rates.notes,
        ),
        Result(
            "compute_seconds",
            roofline.compute_seconds(args.flops, peak),
            "s",
            "F/Π",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=rates.notes,
        ),
        Result(
            "memory_seconds",
            roofline.memory_seconds(args.bytes, bw),
            "s",
            "R/β",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=rates.notes,
        ),
    ]
    if args.bytes > 0:
        out.append(
            Result(
                "arithmetic_intensity",
                roofline.arithmetic_intensity(args.flops, args.bytes),
                "FLOP/B",
                "F/R",
                inputs,
                A_CH1_TIME,
            )
        )
    out.append(
        Result(
            "ridge_point",
            roofline.ridge_point(peak, bw),
            "FLOP/B",
            "Π/β",
            inputs,
            A_CH1_TIME,
            notes=rates.notes,
        )
    )
    return out


def _serving(args: argparse.Namespace) -> list[Result]:
    rates = _rates(args, "--peak-tflops, --bandwidth и --memory")
    peak, bw, dev = rates.peak, rates.bandwidth, rates.device
    memory_notes: list[str] = []
    if args.memory is not None:
        memory = args.memory
        memory_source = USER
    elif dev is None:
        raise ValueError(
            "укажите --memory (байт памяти одного устройства) или --device"
        )
    elif dev.memory_bytes is None:
        raise ValueError(
            f"у {dev.id} ёмкость памяти не опубликована в снимке {hardware.SNAPSHOT}; "
            "передайте --memory"
        )
    else:
        memory = dev.memory_bytes
        memory_source = dev.id
        memory_notes.append(
            "номинальная ёмкость по этикетке, а не доступный среде выполнения объём; "
            "резерв среды выполнения задайте через --reserve"
        )
        if dev.scope != SINGLE:
            memory_notes.append(
                _aggregate_note(dev, "ёмкость памяти")
                + "; веса и KV считаются размещёнными во всём агрегате"
            )
        if dev.shared_with_cpu:
            memory_notes.append(
                "память общая с CPU и ОС: модели доступна только её часть"
            )
    kv_request = args.kv_per_token * args.context
    memory_inputs = {
        "memory": memory,
        "memory_source": memory_source,
        "weights": args.weights,
        "kv_request": kv_request,
        "reserve": args.reserve,
    }
    step = serving.tpot_lower_bound_seconds(
        args.batch, args.decode_flops, args.weight_read, kv_request, peak, bw
    )
    step_inputs = {
        "batch": args.batch,
        "decode_flops": args.decode_flops,
        "weight_read": args.weight_read,
        "kv_read_per_request": kv_request,
        **rates.inputs,
    }
    throughput = serving.tokens_per_second(args.batch, step)
    out = [
        Result(
            "max_concurrent_requests",
            serving.max_concurrent_requests(
                memory, args.weights, kv_request, args.reserve
            ),
            "",
            "floor((C − M_w − reserve) / (kv_per_token × context))",
            memory_inputs,
            A_CH8_MEM,
            notes=tuple(memory_notes),
        ),
        Result(
            "tpot_lower_bound_seconds",
            step,
            "s",
            "max(B·F_decode/Π, (R_W + B·R_KV)/β)",
            step_inputs,
            A_CH1_TIME,
            bound="lower",
            notes=rates.notes,
        ),
        Result(
            "tokens_per_second_upper_bound",
            throughput,
            "tok/s",
            "B / T_step",
            step_inputs,
            A_CH1_TIME,
            bound="upper",
            notes=rates.notes,
        ),
    ]
    if args.prefill_flops is not None:
        ttft = serving.ttft_lower_bound_seconds(
            args.prefill_flops, args.weights, peak, bw
        )
        out.append(
            Result(
                "ttft_lower_bound_seconds",
                ttft,
                "s",
                "max(F_prefill/Π, M_w/β)",
                {
                    "prefill_flops": args.prefill_flops,
                    "weights": args.weights,
                    **rates.inputs,
                },
                A_CH1_REF,
                bound="lower",
                notes=rates.notes,
            )
        )
    if args.price_per_hour is not None:
        price = serving.cost_per_million_tokens(args.price_per_hour, throughput)
        out.append(
            Result(
                "cost_per_million_tokens_lower_bound",
                price,
                "$",
                "price_per_hour / 3600 / tok_s × 10^6",
                {
                    "price_per_hour": args.price_per_hour,
                    "tokens_per_second": throughput,
                },
                A_CH3_PRICE,
                bound="lower",
            )
        )
    return out


def _training(args: argparse.Namespace) -> list[Result]:
    spec = _load_spec(args.config)
    wrapper = _wrapper_notes(spec)
    _, _, total = flops.training_matrix_flops(spec, args.tokens)
    params = accounting.parameter_count(spec)
    base = {"model_type": spec.model_type, "tokens": args.tokens}
    out = [
        Result(
            "training_flops_per_sequence",
            total,
            "FLOP",
            "3 × forward, словарная голова на всех токенах",
            base,
            A_CH3_TRAIN,
            notes=wrapper,
        ),
        Result(
            "six_nd_flops_per_sequence",
            flops.six_nd_flops(params, args.tokens),
            "FLOP",
            "6 × N × D",
            base | {"parameters": params},
            A_CH3_TRAIN,
            notes=wrapper,
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
                Result(
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
        require_positive("--total-tokens", args.total_tokens)
        rates = _rates(args, "--peak-tflops", need_bandwidth=False)
        whole = total * args.total_tokens / args.tokens
        seconds = training.training_seconds(whole, args.devices, rates.peak, args.mfu)
        inputs = {
            "total_flops": whole,
            "total_tokens": args.total_tokens,
            "devices": args.devices,
            "mfu": args.mfu,
            **rates.inputs,
        }
        note = f"FLOPs последовательности длиной {args.tokens} линейно масштабированы на все токены"
        out.append(
            Result(
                "training_seconds",
                seconds,
                "s",
                "F_total / (N · Π · MFU)",
                inputs,
                A_CH7_TIME,
                notes=(note, *rates.notes, *wrapper),
            )
        )
    return out


def _checkpoint(args: argparse.Namespace) -> list[Result]:
    require_non_negative("--checkpoint-bytes", args.checkpoint_bytes)
    require_positive("--save-bandwidth", args.save_bandwidth)
    require_int_at_least("--devices", args.devices, 1)
    require_positive("--device-mtbf", args.device_mtbf)
    save = args.checkpoint_bytes / args.save_bandwidth
    rate = args.devices / args.device_mtbf
    inputs = {"save_seconds": save, "failure_rate": rate, "recovery": args.recovery}
    first = training.checkpoint_optimal_interval(save, rate)
    interval = first if args.interval is None else args.interval
    loss_notes = (
        ()
        if args.interval is not None
        else ("τ — оптимум первого порядка; другой интервал задайте --interval",)
    )
    out = [
        Result(
            "first_order_optimal_interval_seconds",
            first,
            "s",
            "sqrt(2c/λ)",
            inputs,
            A_CH10_CKPT,
        ),
        Result(
            "poisson_optimal_interval_seconds",
            training.checkpoint_poisson_optimal_interval(save, rate),
            "s",
            "y − 1 + exp(−y − λc) = 0, τ = y/λ",
            inputs,
            A_CH10_CKPT,
        ),
        Result(
            "first_order_loss_at_interval",
            training.checkpoint_first_order_loss(interval, save, rate, args.recovery),
            "",
            "c/τ + λτ/2 + λr",
            inputs | {"interval": interval},
            A_CH10_CKPT,
            notes=loss_notes,
        ),
    ]
    if (args.stages is None) != (args.microbatches is None):
        raise ValueError(
            "для утилизации конвейера нужны оба значения: --stages и --microbatches"
        )
    if args.stages is not None:
        utilization = training.pipeline_utilization(args.stages, args.microbatches)
        out.append(
            Result(
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


def _speculative(args: argparse.Namespace) -> list[Result]:
    tokens = speculative.expected_tokens_per_round(args.acceptance, args.draft)
    inputs = {"acceptance": args.acceptance, "draft": args.draft}
    return [
        Result(
            "expected_tokens_per_round",
            tokens,
            "tok",
            "Σ a^i, i = 0..k",
            inputs,
            A_CH8_SPEC,
            notes=("позиции черновика принимаются независимо",),
        ),
        Result(
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


def _ring(args: argparse.Namespace) -> list[Result]:
    inputs = {
        "devices": args.devices,
        "message": args.message,
        "bandwidth": args.bandwidth,
        "alpha": args.alpha,
    }
    return [
        Result(
            "ring_allreduce_seconds",
            collectives.ring_allreduce_seconds(
                args.devices, args.message, args.bandwidth, args.alpha
            ),
            "s",
            "2(n−1)α + 2(n−1)M/(nB)",
            inputs,
            A_CH6_RING,
        ),
        Result(
            "bytes_sent_per_device",
            collectives.ring_bytes_sent_per_device(args.devices, args.message),
            "B",
            "2(n−1)M/n",
            inputs,
            A_CH6_RING,
        ),
    ]


def _cost(args: argparse.Namespace) -> list[Result]:
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
    require_positive("--calls", args.calls)
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
        Result(
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
        Result(
            "cost_per_accepted_task",
            cost.cost_per_accepted_task(per_call * args.calls, args.success),
            "$",
            "calls × C_call / p_success",
            {"cost_per_call": per_call, "calls": args.calls, "success": args.success},
            A_CH11_TASK,
        ),
    ]


def _edge(args: argparse.Namespace) -> list[Result]:
    up = units.megabits_per_second_to_bytes(args.up_mbps)
    down = units.megabits_per_second_to_bytes(args.down_mbps)
    upload = units.parse_bytes(args.upload)
    download = units.parse_bytes(args.download)
    inputs = {
        "upload": args.upload,
        "download": args.download,
        "up_mbps": args.up_mbps,
        "down_mbps": args.down_mbps,
        "rtt": args.rtt,
        "compute": args.compute,
    }
    seconds = edge.serial_seconds(upload, up, args.rtt, args.compute, download, down)
    return [
        Result(
            "serial_seconds",
            seconds,
            "s",
            "S_u/B_u + R + T_c + S_d/B_d",
            inputs,
            A_CH12_EDGE,
        )
    ]


def _peak_label(peak: dict[str, Any]) -> str:
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
    scope: tuple[str, ...] = ()
    if dev.scope != SINGLE:
        note = _aggregate_note(dev, "ёмкость, пропускная способность и пики")
        scope = (note + "; на одно устройство снимок их не делит",)
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
        Result(
            "device_count",
            dev.device_count,
            "",
            "spec_scope и gpu_count/npu_count снимка",
            info,
            A_CH1_KEYS,
            notes=scope,
        ),
        Result(
            "memory_bytes",
            dev.memory_bytes,
            "B",
            "nominal_capacity × единица ёмкости",
            info,
            A_CH1_KEYS,
            notes=tuple(memory_notes),
        ),
        Result(
            "bandwidth",
            dev.bandwidth,
            "B/s",
            "bandwidth_bytes_per_second",
            info,
            A_CH1_KEYS,
            notes=tuple(bandwidth_notes),
        ),
        Result(
            "peak_flops",
            peak,
            "FLOP/s",
            "BF16 / накопление FP32 / tensor / dense; все пики снимка — в примечаниях (вход/накопление/блок/sparsity)",
            info,
            A_CH1_KEYS,
            notes=tuple(peak_notes),
        ),
    ]


# --- разбор аргументов ---------------------------------------------------------


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
    fmt = argparse.ArgumentParser(add_help=False)
    fmt.add_argument(
        "--format",
        choices=("json", "md"),
        default=argparse.SUPPRESS,
        help="вывод: md (по умолчанию) или json",
    )
    parser = argparse.ArgumentParser(
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
        name: str, handler: Callable[[argparse.Namespace], list[Result]], help_text: str
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

    p = command("roofline", _roofline, "нижняя граница времени шага: max(F/Π, R/β)")
    device_options(p)
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
    p.add_argument("--kv-per-token", type=float, required=True, help="байт KV на токен")
    p.add_argument(
        "--context", type=int, required=True, help="длина контекста, токенов"
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--memory", type=float, help="байт памяти одного устройства")
    p.add_argument(
        "--reserve", type=float, default=0, help="байт резерва среды выполнения"
    )
    p.add_argument("--price-per-hour", type=float, help="$ за час устройства")

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
    p.add_argument("--stages", type=int, help="стадий конвейера")
    p.add_argument("--microbatches", type=int, help="микропакетов на шаг")

    p = command(
        "speculative",
        _speculative,
        "спекулятивное декодирование: E[N] и безубыточный раунд",
    )
    p.add_argument(
        "--acceptance", type=float, required=True, help="доля принятия позиции, [0, 1]"
    )
    p.add_argument(
        "--draft", type=int, required=True, help="токенов черновика за раунд"
    )
    p.add_argument(
        "--plain-step", type=float, required=True, help="секунд на обычный шаг decode"
    )

    p = command("ring", _ring, "кольцевой AllReduce")
    p.add_argument("--devices", type=int, required=True)
    p.add_argument("--message", type=float, required=True, help="байт")
    p.add_argument(
        "--bandwidth", type=float, required=True, help="байт/с в одном направлении"
    )
    p.add_argument("--alpha", type=float, required=True, help="секунд на раунд")

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        # argparse уже напечатал справку или ошибку разбора
        return stop.code if isinstance(stop.code, int) else 2
    try:
        results = args.handler(args)
    except (ValueError, KeyError) as error:
        message = _error_text(error)
        if args.format == "json":
            print(
                json.dumps(
                    {"command": args.command, "error": message}, ensure_ascii=False
                )
            )
        else:
            print(f"Ошибка: {message}")
        return 2
    snapshot = hardware.SNAPSHOT if getattr(args, "device", None) else None
    if args.format == "json":
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
