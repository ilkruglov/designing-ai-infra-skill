import math
import unittest
from dataclasses import replace
from pathlib import Path

from infra_calc import accounting, model

ANCHORS = (
    "references/source-book/chapter1.md:197",
    "references/source-book/chapter1.md:458",
    "references/source-book/chapter2.md:236",
    "references/source-book/chapter2.md:418",
    "references/source-book/chapter2.md:542",
    "references/source-book/chapter3.md:611",
    "references/source-book/chapter6.md:148",
    "references/source-book/chapter6.md:408",
    "references/source-book/chapter8.md:52",
    "calculations/results/kv-comparison-n8192-b1.json#sha256=1a45a55edf05115855d798572a84cec00dddee2418589e2fdc3aed968441a257",
    "calculations/results/chapter2-model-comparison.json#sha256=8d8192a345eb560c3642fc93b4f3ece38740059b09642f266b352d8fc8c1a83e",
    "calculations/results/qwen3-30b-a3b-decode-b1-s8192.json#sha256=fbf0b07f78bd8d7ab3765f6fc9ad5f6992cc95192449c2f8f1ee0fd01b5bc75b",
    "calculations/results/v3-forward-b64.json#sha256=91bf7eff7450033791c452788a1b2a249665cb263df6e131f823e77976d00598",
    "calculations/results/qwen3-30b-a3b-decode-b64-s8192-balanced.json#sha256=110e9a83aebd7e326ec87c3a652f82477b1b4e7d838a7e56e1eceb1bd6d89700",
)
CONFIGS = Path(__file__).resolve().parent / "fixtures" / "configs"


def spec(name: str) -> model.ModelSpec:
    return model.parse_spec(model.load_config(CONFIGS / f"{name}.json"))


class DenseTest(unittest.TestCase):
    def test_qwen3_8b_reference_numbers(self) -> None:
        s = spec("qwen3-8b")
        # chapter2-model-comparison.json: Qwen3-8B total_parameters
        self.assertEqual(accounting.parameter_count(s), 8_190_735_360)
        # chapter1.md:458: веса BF16 16.38 GB
        self.assertEqual(accounting.weight_bytes(s), 16_381_470_720)
        # chapter8.md:52: 2 × 36 × 8 × 128 × 2 = 147 456 байт
        self.assertEqual(accounting.kv_bytes_per_token(s), 147_456)
        # chapter2.md:236 (пример 2-2): чтение общих весов за шаг decode
        self.assertEqual(accounting.decode_weight_read_bytes(s), 15_136_811_008)
        # kv-comparison-n8192-b1.json: global_history_bytes
        self.assertEqual(accounting.kv_resident_bytes(s, 8192), 1_207_959_552)
        # chapter2-model-comparison.json: Qwen3-8B routed_experts 0 — активны все
        self.assertEqual(
            accounting.parameter_count(s, active=True), accounting.parameter_count(s)
        )
        # таблица эмбеддингов 151 936 × 4096, исключается из чтения за шаг decode
        self.assertEqual(accounting.embedding_parameters(s), 622_329_856)

    def test_tied_embeddings_head_is_still_read_every_decode_step(self) -> None:
        # chapter2.md:236: словарная голова читает всю матрицу на каждом шаге
        # decode; при общих весах это та же таблица, чтение не меняется
        tied = replace(spec("qwen3-8b"), tied_embeddings=True)
        self.assertEqual(accounting.decode_weight_read_bytes(tied), 15_136_811_008)

    def test_prefill_reads_only_input_rows_of_embedding_table(self) -> None:
        # вывод вручную: без общих весов prefill n токенов читает не больше
        # min(V, n) строк таблицы 151 936 × 4096, словарную голову — целиком (логиты
        # хотя бы последнего токена), слои — все. При n = 16:
        # 8 190 735 360 − (151 936 − 16) × 4096 = 7 568 471 040 параметров;
        # × 2 B = 15 136 942 080 = decode_weight_read_bytes + 16 × 4096 × 2
        s = spec("qwen3-8b")
        self.assertEqual(
            accounting.prefill_weight_read_parameters(s, 16), 7_568_471_040
        )
        self.assertEqual(
            2 * accounting.prefill_weight_read_parameters(s, 16),
            accounting.decode_weight_read_bytes(s) + 16 * 4096 * 2,
        )
        # n ≥ V: вся таблица, чтение = все параметры
        self.assertEqual(
            accounting.prefill_weight_read_parameters(s, 151_936),
            accounting.parameter_count(s),
        )
        # общий префикс может закрыть весь вход: строк таблицы — ноль
        self.assertEqual(
            accounting.prefill_weight_read_parameters(s, 0),
            8_190_735_360 - 622_329_856,
        )
        # общие веса: голова читает всю матрицу, она же — таблица; чтение = все
        # параметры 7 568 405 504 (без отдельной головы) при любом n
        tied = replace(s, tied_embeddings=True)
        for tokens in (0, 16):
            with self.subTest(tokens=tokens):
                self.assertEqual(
                    accounting.prefill_weight_read_parameters(tied, tokens),
                    7_568_405_504,
                )
        with self.assertRaises(ValueError):
            accounting.prefill_weight_read_parameters(s, -1)

    def test_negative_context_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            accounting.kv_resident_bytes(spec("qwen3-8b"), -1)
        self.assertEqual(accounting.kv_resident_bytes(spec("qwen3-8b"), 0), 0)

    def test_fractional_context_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            accounting.kv_resident_bytes(spec("qwen3-8b"), 2.5)

    def test_llama_70b(self) -> None:
        s = spec("deepseek-r1-distill-llama-70b")
        # chapter1.md:197: веса BF16 141.11 GB
        self.assertEqual(round(accounting.weight_bytes(s) / 1e9, 2), 141.11)
        # kv-comparison-n8192-b1.json: deepseek-r1-distill-llama-70b,
        # global_growth_bytes_per_token_per_request
        self.assertEqual(accounting.kv_bytes_per_token(s), 327_680)


class MoeTest(unittest.TestCase):
    def test_qwen3_30b_a3b(self) -> None:
        s = spec("qwen3-30b-a3b")
        # qwen3-30b-a3b-decode-b1-s8192.json: parameters
        self.assertEqual(accounting.parameter_count(s), 30_532_122_624)
        # parameters − routed_expert_total + routed_expert_active_per_token
        self.assertEqual(
            accounting.parameter_count(s, active=True),
            30_532_122_624 - 28_991_029_248 + 1_811_939_328,
        )
        # qwen3-30b-a3b-decode-b1-s8192.json: kv_bytes_per_token_per_request;
        # kv-comparison-n8192-b1.json: qwen3-30b-a3b
        self.assertEqual(accounting.kv_bytes_per_token(s), 98_304)

    def test_prefill_read_is_one_token_experts_plus_input_rows(self) -> None:
        # вывод вручную: активные параметры одного токена (U = k) без непрочитанных
        # строк таблицы. Qwen3-30B-A3B, V = 151 936, h = 2048, n = 16:
        # 3 353 032 704 − (151 936 − 16) × 2048 = 3 041 900 544;
        # × 2 B = 6 083 801 088 = 6 083 735 552 (decode) + 16 × 2048 × 2.
        # DeepSeek-V3, V = 129 280, h = 7168, n = 16:
        # 37 552 297 472 − (129 280 − 16) × 7168 = 36 625 733 120
        cases = (
            ("qwen3-30b-a3b", 3_041_900_544, 16 * 2048),
            ("deepseek-v3", 36_625_733_120, 16 * 7168),
        )
        for name, expected, rows in cases:
            with self.subTest(name=name):
                s = spec(name)
                read = accounting.prefill_weight_read_parameters(s, 16)
                self.assertEqual(read, expected)
                self.assertEqual(
                    2 * read, accounting.decode_weight_read_bytes(s) + 2 * rows
                )


class MlaTest(unittest.TestCase):
    def test_deepseek_v3(self) -> None:
        s = spec("deepseek-v3")
        # v3-forward-b64.json: logical_base_parameters
        self.assertEqual(accounting.parameter_count(s), 671_026_419_200)
        # chapter3.md:611 (строка 631): около 37B активных параметров на токен
        self.assertEqual(int(accounting.parameter_count(s, active=True) / 1e9), 37)
        # kv-comparison: 61 × (512 + 64) × 2 байта
        self.assertEqual(accounting.kv_bytes_per_token(s), 70_272)


class HybridTest(unittest.TestCase):
    def test_qwen35_state(self) -> None:
        s = spec("qwen3.5-397b-a17b")
        # kv-comparison: 15 слоёв полного внимания, фиксированное состояние 45 линейных
        self.assertEqual(accounting.kv_bytes_per_token(s), 30_720)
        self.assertEqual(accounting.fixed_state_bytes(s), (188_743_680, 4_423_680))

    def test_hybrid_parameters_are_not_guessed(self) -> None:
        # chapter2.md:418: у гибридной модели формулы параметров нет — отказ с полем
        # и подсказкой, где взять число (карточка модели, calc.py model --params)
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            accounting.parameter_count(spec("qwen3.5-397b-a17b"))
        self.assertEqual(caught.exception.fields, ["linear_attention"])
        self.assertEqual(caught.exception.model_type, "qwen3_5_moe_text")
        self.assertIn("calc.py model --params", str(caught.exception))


class TensorParallelTest(unittest.TestCase):
    def test_gqa_kv_split_and_duplication_book(self) -> None:
        # chapter6.md:224: «У Qwen3-32B есть 8 KV-голов, поэтому при TP8 на каждую карту
        # приходится по одной. При TP16 ... обе карты должны хранить состояние этой
        # KV-головы. При контексте 128K состояние каждой KV-головы занимает 4 GiB:
        # суммарно 32 GiB на восьми картах и 64 GiB на шестнадцати.»
        s = spec("qwen3-32b")
        per_token = accounting.kv_bytes_per_token(s)
        for tp, total_gib in ((8, 32), (16, 64)):
            with self.subTest(tp=tp):
                per_card = per_token * 131_072 // accounting.tp_kv_divisor(s, tp)
                self.assertEqual(per_card, 4 * 2**30)
                self.assertEqual(per_card * tp, total_gib * 2**30)

    def test_gqa_divisor_below_kv_heads(self) -> None:
        # вывод вручную: при TP ≤ числа голов KV каждая карта хранит kv_heads/TP голов
        s = spec("qwen3-30b-a3b")  # 4 головы KV
        self.assertEqual(accounting.tp_kv_divisor(s, 1), 1)
        self.assertEqual(accounting.tp_kv_divisor(s, 2), 2)
        self.assertEqual(accounting.tp_kv_divisor(s, 8), 4)

    def test_mla_latent_is_not_split(self) -> None:
        # chapter2.md:334: MLA хранит на токен одну скрытую переменную до повышающей
        # проекции, общую для всех голов; chapter6.md:224: TP делит внимание по головам —
        # поэтому каждая карта TP хранит латентный KV целиком
        s = spec("deepseek-v3")
        self.assertEqual(accounting.tp_kv_divisor(s, 8), 1)

    def test_hybrid_state_split(self) -> None:
        # chapter6.md:171: «Число запросов 128K/32K на двух H100 | 2/10 | 7/29 | 31/117»;
        # 31/117 у Qwen3.6 получается, только если фиксированное состояние линейных
        # слоёв делится между двумя картами (проверка в test_cli)
        s = spec("qwen3.6-35b-a3b")
        self.assertEqual(accounting.tp_kv_divisor(s, 2), 2)
        self.assertEqual(accounting.tp_fixed_state_divisor(s, 2), 2)
        self.assertEqual(accounting.tp_fixed_state_divisor(spec("qwen3-8b"), 4), 1)

    def test_tp_must_divide_heads(self) -> None:
        # chapter6.md:224: «Голову нельзя разделить дальше: это минимальная единица
        # распределения»
        for name, tp in (("qwen3-8b", 3), ("qwen3-32b", 6), ("qwen3-30b-a3b", 0)):
            with self.subTest(name=name, tp=tp), self.assertRaises(ValueError):
                accounting.tp_kv_divisor(spec(name), tp)
        # вывод вручную: TP = 32 делит 64 головы, каждая из 8 голов KV хранится на
        # 4 картах — делитель 8; TP = 64 не делит 16 голов внимания Qwen3.6
        self.assertEqual(accounting.tp_kv_divisor(spec("qwen3-32b"), 32), 8)
        with self.assertRaises(ValueError):
            accounting.tp_fixed_state_divisor(spec("qwen3.6-35b-a3b"), 64)


class ExpertUnionTest(unittest.TestCase):
    def test_expected_experts_book(self) -> None:
        # chapter6.md:439, формула (6-8): E[E_active] = E[1 − (1 − k/E)^m];
        # chapter6.md:442: «При $E=128,k=8,m=8$ ожидается около 52 активных экспертов, а
        # идеальный объём чтения составляет приблизительно 1,8 GiB» (36 MiB на эксперта)
        experts = accounting.expected_active_experts(128, 8, 8)
        self.assertEqual(round(experts), 52)
        self.assertEqual(round(experts * 36 * 2**20 / 2**30, 1), 1.8)
        # шаблон sizing-sheet: U(4) = 128·[1 − (1 − 8/128)^4] ≈ 29.123
        self.assertEqual(
            round(accounting.expected_active_experts(128, 8, 4), 3), 29.123
        )
        # один токен выбирает ровно k экспертов
        self.assertEqual(accounting.expected_active_experts(128, 8, 1), 8)

    def test_batch_weight_read_matches_author_balanced(self) -> None:
        # qwen3-30b-a3b-decode-b64-s8192-balanced.json: expert_union_per_layer 128,
        # weight_read_once_per_operator_bytes 60 442 177 536; автор добавляет 64 строки
        # эмбеддингов по 2048 × 2 байта, которые decode_weight_read_bytes не считает;
        # chapter6.md:167: «Чтение весов за шаг (batch 64, контекст 32K) | ... | 60,44 GB»
        s = spec("qwen3-30b-a3b")
        read = accounting.batch_decode_weight_read_bytes(s, 128)
        self.assertEqual(read, 60_442_177_536 - 64 * 2048 * 2)
        self.assertEqual(round(read / 1e9, 2), 60.44)
        # calc.py forward --model qwen3-30b-a3b --batch 4 --history 8191 --tokens 1 --routing balanced
        # (код автора на 3bdcb4fc): expert_union_per_layer 32,
        # weight_read_once_per_operator_bytes 16 955 387 904 — минус 4 строки эмбеддингов
        self.assertEqual(
            accounting.batch_decode_weight_read_bytes(s, 32),
            16_955_387_904 - 4 * 2048 * 2,
        )

    def test_one_token_equals_single_request_read(self) -> None:
        # U = k: объединение экспертов одного токена — те же k экспертов
        for name in ("qwen3-30b-a3b", "deepseek-v3"):
            with self.subTest(name=name):
                s = spec(name)
                self.assertEqual(
                    accounting.batch_decode_weight_read_bytes(s, s.experts_per_token),
                    accounting.decode_weight_read_bytes(s),
                )

    def test_fractional_expected_union(self) -> None:
        # вывод вручную: non_expert + U·layers·3·h·f·2 при U = 29.123 для qwen3-30b-a3b:
        # 2 459 856 896 + 29.1233·48·9 437 184 байт
        s = spec("qwen3-30b-a3b")
        union = accounting.expected_active_experts(128, 8, 4)
        self.assertEqual(
            accounting.batch_decode_weight_read_bytes(s, union),
            int(2_459_856_896 + union * 48 * 3 * 2048 * 768 * 2),
        )

    def test_rejects_impossible_union_and_dense(self) -> None:
        s = spec("qwen3-30b-a3b")
        for union in (7, 129, -1, math.nan):
            with self.subTest(union=union), self.assertRaises(ValueError):
                accounting.batch_decode_weight_read_bytes(s, union)
        with self.assertRaises(ValueError):
            accounting.batch_decode_weight_read_bytes(spec("qwen3-8b"), 8)
        for args in ((128, 0, 4), (128, 129, 4), (128, 8, 0), (0, 8, 4)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                accounting.expected_active_experts(*args)


class WindowTest(unittest.TestCase):
    def test_window_caps_resident_kv(self) -> None:
        s = model.ModelSpec(
            model_type="mistral",
            family="dense",
            layers=2,
            hidden=64,
            heads=4,
            kv_heads=2,
            head_dim=16,
            vocab=100,
            tied_embeddings=False,
            intermediate=128,
            window=1024,
        )
        per_token = accounting.kv_bytes_per_token(s)
        self.assertEqual(accounting.kv_resident_bytes(s, 4096), per_token * 1024)
        self.assertEqual(accounting.kv_resident_bytes(s, 512), per_token * 512)


if __name__ == "__main__":
    unittest.main()
