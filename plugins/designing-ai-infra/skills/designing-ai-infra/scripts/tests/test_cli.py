import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from infra_calc import cli
from infra_calc.result import Result

ANCHORS = (
    "references/source-book/chapter1.md:152",
    "references/source-book/chapter1.md:263",
    "references/source-book/chapter1.md:450",
    "references/source-book/chapter3.md:384",
    "references/source-book/chapter3.md:442",
    "references/source-book/chapter6.md:473",
    "references/source-book/chapter6.md:705",
    "references/source-book/chapter8.md:536",
    "references/source-book/chapter10.md:151",
    "references/source-book/chapter10.md:553",
    "references/source-book/chapter11.md:515",
    "references/source-book/chapter12.md:11",
    "calculations/results/checkpoint-interval-book.json#sha256=e986d06ba47b7d0c13b54a99cc2abde1744d674eb470b26d9cf00e417ac0c6ad",
)
SCRIPTS = Path(__file__).resolve().parents[1]
SKILL = SCRIPTS.parent
CONFIGS = SCRIPTS / "tests" / "fixtures" / "configs"
QWEN3_8B = str(CONFIGS / "qwen3-8b.json")
AUTHOR = json.loads(
    (SCRIPTS / "tests" / "fixtures" / "author_results.json").read_text()
)["checkpoint-interval-book"]


def call(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        code = cli.main(list(argv))
    return code, out.getvalue()


def run(*argv: str) -> dict[str, Any]:
    code, out = call(*argv, "--format", "json")
    assert code == 0, out
    return json.loads(out)


def values(*argv: str) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in run(*argv)["results"]}


def fails(*argv: str) -> str:
    code, out = call(*argv, "--format", "json")
    assert code == 2, out
    return json.loads(out)["error"]


class ModelCommandTest(unittest.TestCase):
    def test_qwen3_8b_reference_numbers(self) -> None:
        v = values("model", "--config", QWEN3_8B, "--context", "2048")
        # chapter3.md:414: «рассмотрим 8 190 735 360 параметров Qwen3-8B»
        self.assertEqual(v["parameters"]["value"], 8_190_735_360)
        # chapter1.md:456: «| Полные веса BF16 | 16.38 GB |»
        self.assertEqual(round(v["weight_bytes"]["value"] / 1e9, 2), 16.38)
        # chapter1.md:457: «Полный KV модели на каждый токен контекста | 144 KiB»
        self.assertEqual(v["kv_bytes_per_token"]["value"], 144 * 2**10)
        # chapter1.md:458: «| KV контекста из 2048 токенов | 288 MiB |»
        self.assertEqual(v["kv_resident_bytes"]["value"], 288 * 2**20)
        # chapter1.md:459: «Матричные операции prefill для 2048 токенов | 29.69 TFLOPs»
        self.assertEqual(round(v["prefill_flops"]["value"] / 1e12, 2), 29.69)
        # chapter1.md:460: «Матричные операции одного шага decode | 16.34 GFLOPs»
        self.assertEqual(round(v["decode_step_flops"]["value"] / 1e9, 2), 16.34)
        # chapter1.md:461: «Основной объём чтения весов за один шаг decode | 15.14 GB»
        self.assertEqual(round(v["decode_weight_read_bytes"]["value"] / 1e9, 2), 15.14)
        for item in v.values():
            self.assertTrue(item["anchor"].startswith("references/source-book/"))

    def test_unsupported_model_exits_with_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "config.json"
            bad.write_text(
                json.dumps({"model_type": "kimi_linear", "linear_attn_config": {}})
            )
            message = fails("model", "--config", str(bad))
        self.assertIn("k3-forward", message)

    def test_missing_config_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            message = fails("model", "--config", str(Path(tmp) / "absent.json"))
        self.assertIn("absent.json", message)

    def test_config_without_required_field_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "config.json"
            bad.write_text(json.dumps({"model_type": "qwen3"}))
            message = fails("model", "--config", str(bad))
        self.assertIn("num_attention_heads", message)

    def test_hybrid_parameters_are_refused_not_guessed(self) -> None:
        config = str(CONFIGS / "qwen3.5-397b-a17b.json")
        v = values("model", "--config", config)
        self.assertIsNone(v["parameters"]["value"])
        notes = " ".join(v["parameters"]["notes"])
        self.assertIn("--params", notes)
        # обёртка qwen3_5_moe над текстовой моделью: энкодеры не учтены
        self.assertIn("qwen3_5_moe", notes)
        self.assertIn("энкодер", notes)
        self.assertNotIn("weight_bytes", v)

    def test_hybrid_parameters_from_user_are_labelled(self) -> None:
        # число из карточки модели передаётся как есть; значение условное
        config = str(CONFIGS / "qwen3.5-397b-a17b.json")
        v = values("model", "--config", config, "--params", "1000")
        self.assertEqual(v["parameters"]["value"], 1000)
        self.assertIn("задано пользователем", v["parameters"]["formula"])
        self.assertEqual(v["weight_bytes"]["value"], 2000)

    def test_params_for_countable_model_is_refused(self) -> None:
        message = fails("model", "--config", QWEN3_8B, "--params", "1000")
        self.assertIn("8190735360", message)


class RooflineCommandTest(unittest.TestCase):
    def test_roofline_uses_snapshot_and_names_it(self) -> None:
        data = run(
            "roofline", "--device", "h100-sxm", "--flops", "140e9", "--bytes", "70e9"
        )
        v = {item["name"]: item for item in data["results"]}
        # chapter1.md:268: «\frac{70\ \mathrm{GB}}{3350\ \mathrm{GB/s}}\approx20{,}90\ \mathrm{ms}»
        self.assertEqual(round(v["step_lower_bound_seconds"]["value"] * 1e3, 2), 20.90)
        # chapter1.md:268: «\frac{140\ \mathrm{GFLOPs}}{989400\ \mathrm{GFLOP/s}}\approx0{,}1415\ \mathrm{ms}»
        self.assertEqual(round(v["compute_seconds"]["value"] * 1e3, 4), 0.1415)
        self.assertEqual(v["step_lower_bound_seconds"]["bound"], "lower")
        self.assertIn("56ecb425", data["hardware_snapshot"])

    def test_format_before_command(self) -> None:
        code, out = call(
            "--format",
            "json",
            "roofline",
            "--peak-tflops",
            "989.4",
            "--bandwidth",
            "3.35e12",
            "--flops",
            "140e9",
            "--bytes",
            "70e9",
        )
        self.assertEqual(code, 0, out)
        data = json.loads(out)
        # chapter1.md:268: 20,90 ms без снимка — пик и полоса заданы явно
        self.assertEqual(round(data["results"][0]["value"] * 1e3, 2), 20.90)
        self.assertIsNone(data["hardware_snapshot"])

    def test_markdown_marks_lower_bound(self) -> None:
        code, out = call(
            "roofline", "--device", "h100-sxm", "--flops", "140e9", "--bytes", "70e9"
        )
        self.assertEqual(code, 0)
        self.assertIn("(нижняя граница)", out)
        self.assertIn("56ecb425", out)
        self.assertIn("references/source-book/chapter1.md:263", out)
        # входы — в том же виде, что и значения, а не 140000000000.0
        self.assertIn("flops=1.4e+11", out)
        self.assertIn("accumulator=FP32", out)

    def test_accumulator_selects_peak(self) -> None:
        # chapter1.md:184: «Пиковая производительность матриц BF16 (TFLOP/s) | 165.2»
        # у RTX 4090 при накоплении FP32; с накоплением FP16 в снимке 330.3
        # (data/hardware.json, как в test_hardware.test_accumulator_defaults_to_fp32)
        args = ("roofline", "--device", "rtx4090", "--precision", "FP16")
        work = ("--flops", "1e12", "--bytes", "1")
        fp32 = values(*args, *work)["compute_seconds"]
        self.assertEqual(fp32["inputs"]["peak"], 165.2e12)
        self.assertEqual(fp32["inputs"]["accumulator"], "FP32")
        fp16 = values(*args, "--accumulator", "FP16", *work)["compute_seconds"]
        self.assertEqual(fp16["inputs"]["peak"], 330.3e12)

    def test_peak_lookup_failure_names_both_ways_out(self) -> None:
        # у b200-sxm в снимке FP16 с накоплением «unspecified», пика FP16/FP32 нет
        message = fails(
            "roofline",
            "--device",
            "b200-sxm",
            "--precision",
            "FP16",
            "--flops",
            "1",
            "--bytes",
            "1",
        )
        self.assertIn("--accumulator", message)
        self.assertIn("--peak-tflops", message)

    def test_aggregate_is_refused_without_opt_in(self) -> None:
        # chapter6.md:711: «GB200 NVL72 объединяет 72 GPU» — стойка, а не карта
        message = fails(
            "roofline", "--device", "gb200-nvl72", "--flops", "1", "--bytes", "1"
        )
        self.assertIn("агрегат", message)
        self.assertIn("--allow-aggregate", message)

    def test_aggregate_with_opt_in_is_noted(self) -> None:
        v = values(
            "roofline",
            "--device",
            "gb200-nvl72",
            "--allow-aggregate",
            "--flops",
            "1",
            "--bytes",
            "1",
        )
        self.assertIn("72", " ".join(v["step_lower_bound_seconds"]["notes"]))

    def test_rates_without_device_are_required(self) -> None:
        message = fails("roofline", "--flops", "1", "--bytes", "1")
        self.assertIn("--device", message)


class ServingCommandTest(unittest.TestCase):
    RTX = (
        "serving",
        "--device",
        "rtx-pro6000-blackwell-ws",
        "--weights",
        "16.38e9",
        "--weight-read",
        "15.14e9",
        "--decode-flops",
        "16.34e9",
        "--prefill-flops",
        "29.69e12",
        "--kv-per-token",
        "147456",
        "--context",
        "2048",
    )

    def test_rtx_pro_6000_bounds(self) -> None:
        # chapter1.md:464: при 503.8 TFLOP/s «матричные операции prefill занимают около
        # 58.9 ms»; при 1.792 TB/s чтение весов и KV за шаг decode «занимает около 8.62 ms»
        v = values(*self.RTX)
        self.assertEqual(round(v["ttft_lower_bound_seconds"]["value"] * 1e3, 1), 58.9)
        self.assertEqual(round(v["tpot_lower_bound_seconds"]["value"] * 1e3, 2), 8.62)
        self.assertEqual(v["tpot_lower_bound_seconds"]["bound"], "lower")
        self.assertEqual(v["tokens_per_second_upper_bound"]["bound"], "upper")

    def test_aggregate_memory_needs_per_device_value(self) -> None:
        message = fails(
            "serving",
            "--device",
            "gb200-nvl72",
            "--weights",
            "1e9",
            "--weight-read",
            "1e9",
            "--decode-flops",
            "1e9",
            "--kv-per-token",
            "1000",
            "--context",
            "10",
        )
        self.assertIn("--memory", message)
        self.assertIn("--allow-aggregate", message)

    def test_aggregate_memory_with_opt_in_is_noted(self) -> None:
        v = values(
            "serving",
            "--device",
            "gb200-nvl72",
            "--allow-aggregate",
            "--weights",
            "1e9",
            "--weight-read",
            "1e9",
            "--decode-flops",
            "1e9",
            "--kv-per-token",
            "1000",
            "--context",
            "10",
        )
        notes = " ".join(v["max_concurrent_requests"]["notes"])
        self.assertIn("72", notes)


class TrainingCommandTest(unittest.TestCase):
    def test_qwen3_8b_sequence_and_zero_states(self) -> None:
        v = values("training", "--config", QWEN3_8B, "--tokens", "8192", "--dp", "8")
        # chapter3.md:452: «| Прямой и обратный проходы | 431,368 |» TFLOPs
        self.assertEqual(
            round(v["training_flops_per_sequence"]["value"] / 1e12, 3), 431.368
        )
        # chapter3.md:454: «| Оценка $6ND$ по общему числу параметров | 402,591 |»
        self.assertEqual(
            round(v["six_nd_flops_per_sequence"]["value"] / 1e12, 3), 402.591
        )
        # chapter10.md:190-193: 122.1 / 42.0 / 28.6 / 15.3 GiB на GPU при восьми GPU
        gib = [
            round(v[f"zero{stage}_state_bytes_per_gpu"]["value"] / 2**30, 1)
            for stage in range(4)
        ]
        self.assertEqual(gib, [122.1, 42.0, 28.6, 15.3])

    def test_duration_needs_all_its_inputs(self) -> None:
        message = fails(
            "training", "--config", QWEN3_8B, "--tokens", "8192", "--devices", "8"
        )
        self.assertIn("--mfu", message)


class OtherCommandsTest(unittest.TestCase):
    def test_checkpoint_interval(self) -> None:
        v = values(
            "checkpoint",
            "--checkpoint-bytes",
            str(AUTHOR["checkpoint_payload_bytes"]),
            "--save-bandwidth",
            str(AUTHOR["save_bandwidth_bytes_per_second"]),
            "--devices",
            str(AUTHOR["devices"]),
            "--device-mtbf",
            str(AUTHOR["device_mtbf_seconds"]),
            "--recovery",
            str(AUTHOR["recovery_seconds"]),
            "--interval",
            "60",
        )
        # checkpoint-interval-book.json: first_order_optimal_useful_interval_seconds,
        # poisson_optimal_useful_interval_seconds, first_order_loss_at_60;
        # chapter10.md:570: «около 965 s»
        self.assertAlmostEqual(
            v["first_order_optimal_interval_seconds"]["value"],
            AUTHOR["first_order_optimal_useful_interval_seconds"],
            places=6,
        )
        self.assertAlmostEqual(
            v["poisson_optimal_interval_seconds"]["value"],
            AUTHOR["poisson_optimal_useful_interval_seconds"],
            places=6,
        )
        self.assertAlmostEqual(
            v["first_order_loss_at_interval"]["value"],
            AUTHOR["first_order_loss_at_60"],
            places=12,
        )

    def test_checkpoint_rejects_zero_bandwidth(self) -> None:
        message = fails(
            "checkpoint",
            "--checkpoint-bytes",
            "1e9",
            "--save-bandwidth",
            "0",
            "--devices",
            "8",
            "--device-mtbf",
            "1e6",
        )
        self.assertIn("--save-bandwidth", message)

    def test_pipeline_utilization(self) -> None:
        # chapter10.md:336: «При $p=4,m=8$ получаем $u=8/11\approx72.7\%$.»
        v = values(
            "checkpoint",
            "--checkpoint-bytes",
            "1e9",
            "--save-bandwidth",
            "1e9",
            "--devices",
            "8",
            "--device-mtbf",
            "1e6",
            "--stages",
            "4",
            "--microbatches",
            "8",
        )
        self.assertEqual(round(v["pipeline_utilization"]["value"] * 100, 1), 72.7)

    def test_speculative(self) -> None:
        # chapter8.md:552: «Для черновика AAAA значение $a=1/4$, а среднее число
        # выходных токенов за раунд составляет около 1,33»; раунд проверки 26,26 мс
        v = values(
            "speculative",
            "--acceptance",
            "0.25",
            "--draft",
            "4",
            "--plain-step",
            "26.26e-3",
        )
        self.assertEqual(round(v["expected_tokens_per_round"]["value"], 2), 1.33)
        # chapter8.md:554: «на запрос черновика остаётся около 8,7 мс для AAAA»
        budget = v["breakeven_round_seconds"]["value"] - 26.26e-3
        self.assertEqual(round(budget * 1e3, 1), 8.7)

    def test_ring(self) -> None:
        # chapter6.md:499-501: M = 10 KiB, B = 450 GB/s, α = 0,822 μs, TP8:
        # «каждая карта отправляет 17,5 KiB, ... а одна операция — около 11,55 μs»
        v = values(
            "ring",
            "--devices",
            "8",
            "--message",
            "10240",
            "--bandwidth",
            "450e9",
            "--alpha",
            "0.822e-6",
        )
        self.assertEqual(round(v["ring_allreduce_seconds"]["value"] * 1e6, 2), 11.55)
        self.assertEqual(v["bytes_sent_per_device"]["value"] / 1024, 17.5)

    def test_cost_example_11_7(self) -> None:
        # chapter11.md:521-532: A — 1 000 новых и 19 000 из кэша по 1 / 0.1 $, 1 800 + 200
        # выходных по 5 $; «стоимость каждого вызова составляет 0,0129 доллара, а средняя
        # стоимость успешно выполненной задачи — $0.0129/0.8\approx0.0161$ доллара»
        v = values(
            "cost",
            "--input-tokens",
            "1000",
            "--output-tokens",
            "2000",
            "--input-price",
            "1",
            "--output-price",
            "5",
            "--cached-tokens",
            "19000",
            "--cache-read-price",
            "0.1",
            "--success",
            "0.8",
        )
        self.assertAlmostEqual(v["cost_per_call"]["value"], 0.0129)
        self.assertEqual(round(v["cost_per_accepted_task"]["value"], 4), 0.0161)

    def test_cost_accepts_expected_tokens(self) -> None:
        # chapter11.md:538: при h = 50% стоимость успешной задачи B «примерно до 0,0264»;
        # ожидаемые токены: 9 500 из кэша и 1 000 + 9 500 обычных, успех 0.98
        v = values(
            "cost",
            "--input-tokens",
            "10500.0",
            "--output-tokens",
            "300",
            "--input-price",
            "2",
            "--output-price",
            "10",
            "--cached-tokens",
            "9500.0",
            "--cache-read-price",
            "0.2",
            "--success",
            "0.98",
        )
        self.assertEqual(round(v["cost_per_accepted_task"]["value"], 4), 0.0264)

    def test_edge(self) -> None:
        # chapter12.md:33-37: «T_{\mathrm{serial}}=\frac{30\times8}{20}+0.1+0.3+\frac{5\times8}{100} =12.8\ \mathrm{s}»
        v = values(
            "edge",
            "--upload",
            "30 MB",
            "--download",
            "5 MB",
            "--up-mbps",
            "20",
            "--down-mbps",
            "100",
            "--rtt",
            "0.1",
            "--compute",
            "0.3",
        )
        self.assertAlmostEqual(v["serial_seconds"]["value"], 12.8)

    def test_edge_rejects_size_without_unit(self) -> None:
        message = fails(
            "edge",
            "--upload",
            "30",
            "--download",
            "5 MB",
            "--up-mbps",
            "20",
            "--down-mbps",
            "100",
            "--rtt",
            "0.1",
            "--compute",
            "0.3",
        )
        self.assertIn("единиц", message)


class DeviceCommandTest(unittest.TestCase):
    def test_single_device(self) -> None:
        # chapter1.md:182-184: H100 SXM — 80 GB, 3.35 TB/s, 989.4 TFLOP/s BF16
        v = values("device", "--device", "h100-sxm")
        self.assertEqual(v["memory_bytes"]["value"], 80e9)
        self.assertEqual(v["bandwidth"]["value"], 3.35e12)
        self.assertEqual(v["device_count"]["value"], 1)
        self.assertEqual(v["device_count"]["inputs"]["scope"], "single_device")
        self.assertEqual(v["peak_flops"]["value"], 989.4e12)
        notes = " ".join(v["peak_flops"]["notes"])
        self.assertIn("BF16/FP32/tensor/dense: 989.4 TFLOP/s", notes)
        # целочисленные пики снимка показаны как TOPS, а не FLOP/s
        self.assertIn("INT8/INT32/tensor/dense", notes)
        self.assertIn("TOPS", notes)

    def test_aggregate_shows_count_and_scope(self) -> None:
        # chapter6.md:711: «GB200 NVL72 объединяет 72 GPU и 36 CPU Grace»
        v = values("device", "--device", "gb200-nvl72")
        self.assertEqual(v["device_count"]["value"], 72)
        self.assertEqual(v["device_count"]["inputs"]["scope"], "gpu_aggregate")
        self.assertIn("72", " ".join(v["memory_bytes"]["notes"]))

    def test_shared_memory_is_flagged(self) -> None:
        v = values("device", "--device", "m2-max-38gpu-96gb")
        self.assertTrue(v["memory_bytes"]["inputs"]["shared_with_cpu"])
        self.assertIn("CPU", " ".join(v["memory_bytes"]["notes"]))

    def test_unknown_device_suggests_ids(self) -> None:
        self.assertIn("h100-sxm", fails("device", "--device", "h100"))


class ResultTest(unittest.TestCase):
    def test_unknown_bound_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            Result("x", 1.0, "s", "F/Π", {}, "a", bound="minimum")

    def test_markdown_of_refused_value(self) -> None:
        text = Result(
            "x", None, "", "не вычисляется", {}, "a", notes=("почему",)
        ).to_markdown()
        self.assertIn("не вычисляется", text)
        self.assertIn("- почему", text)
        self.assertNotIn("входные данные", text)

    def test_markdown_shows_integers_in_full(self) -> None:
        # chapter3.md:414 записывает число параметров как «8 190 735 360»
        text = Result("parameters", 8_190_735_360, "", "Σ", {}, "a").to_markdown()
        self.assertIn("8 190 735 360", text)


class AnchorTest(unittest.TestCase):
    def test_every_result_anchor_is_a_heading(self) -> None:
        commands = (
            ("model", "--config", QWEN3_8B, "--context", "2048"),
            ("model", "--config", str(CONFIGS / "qwen3.5-397b-a17b.json")),
            ("roofline", "--device", "h100-sxm", "--flops", "1", "--bytes", "1"),
            (*ServingCommandTest.RTX, "--price-per-hour", "2"),
            (
                "training",
                "--config",
                QWEN3_8B,
                "--tokens",
                "8192",
                "--dp",
                "8",
                "--total-tokens",
                "1e12",
                "--devices",
                "8",
                "--mfu",
                "0.4",
                "--device",
                "h100-sxm",
            ),
            (
                "checkpoint",
                "--checkpoint-bytes",
                "1e9",
                "--save-bandwidth",
                "1e9",
                "--devices",
                "8",
                "--device-mtbf",
                "1e6",
                "--stages",
                "4",
                "--microbatches",
                "8",
            ),
            (
                "speculative",
                "--acceptance",
                "0.5",
                "--draft",
                "4",
                "--plain-step",
                "0.01",
            ),
            (
                "ring",
                "--devices",
                "8",
                "--message",
                "1",
                "--bandwidth",
                "1",
                "--alpha",
                "0",
            ),
            ("cost", "--input-tokens", "1", "--input-price", "1"),
            (
                "edge",
                "--upload",
                "1 MB",
                "--download",
                "1 MB",
                "--up-mbps",
                "1",
                "--down-mbps",
                "1",
                "--rtt",
                "0",
                "--compute",
                "0",
            ),
            ("device", "--device", "gb200-nvl72"),
        )
        for argv in commands:
            for item in run(*argv)["results"]:
                with self.subTest(command=argv[0], result=item["name"]):
                    path, line = item["anchor"].rsplit(":", 1)
                    lines = (SKILL / path).read_text(encoding="utf-8").splitlines()
                    self.assertTrue(lines[int(line) - 1].startswith("#"))


if __name__ == "__main__":
    unittest.main()
