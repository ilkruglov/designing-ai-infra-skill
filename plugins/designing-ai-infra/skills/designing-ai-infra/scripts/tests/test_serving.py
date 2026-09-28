import unittest

from infra_calc import serving

ANCHORS = (
    "references/source-book/chapter8.md:52",
    "references/source-book/chapter1.md:263",
)


class ServingTest(unittest.TestCase):
    def test_concurrency_from_memory_inequality(self) -> None:
        # chapter8.md:52: M_w + M_KV + M_a + M_u ≤ C; KV длинного запроса 8192+255 токенов
        per_request = 147_456 * (8192 + 255)
        self.assertEqual(
            serving.max_concurrent_requests(96e9, 16_381_470_720, per_request), 63
        )

    def test_weights_do_not_fit(self) -> None:
        # Review Focus 4: 141.11 GB весов не помещаются в 80 GB — ноль, а не отрицательное число
        self.assertEqual(serving.max_concurrent_requests(80e9, 141.11e9, 1e9), 0)

    def test_tpot_bound_batch_eight(self) -> None:
        # chapter1.md:263: восемь запросов делят чтение 70 GB весов
        step = serving.tpot_lower_bound_seconds(8, 140e9, 70e9, 0, 989.4e12, 3.35e12)
        self.assertEqual(round(step * 1e3, 2), 20.90)
        self.assertEqual(round(serving.tokens_per_second(8, step), 1), 382.9)

    def test_ttft_bound(self) -> None:
        self.assertEqual(
            round(
                serving.ttft_lower_bound_seconds(29.69e12, 16.38e9, 989.4e12, 3.35e12)
                * 1e3,
                2,
            ),
            30.01,
        )

    def test_cost_per_million_tokens(self) -> None:
        # 3.6 $/час при 1000 токенов/с: 1 $ за миллион токенов
        self.assertEqual(serving.cost_per_million_tokens(3.6, 1000), 1.0)

    def test_concurrency_rejects_invalid_sizes(self) -> None:
        # нулевой KV на запрос дал бы деление на ноль, отрицательный — отрицательную ёмкость
        for kwargs in (
            {"kv_bytes_per_request": 0},
            {"kv_bytes_per_request": -1.0},
            {"kv_bytes_per_request": 1e9, "reserve_bytes": -1.0},
            {"kv_bytes_per_request": 1e9, "weight_bytes": -1.0},
            {"kv_bytes_per_request": 1e9, "memory_bytes": -1.0},
        ):
            args = {"memory_bytes": 96e9, "weight_bytes": 16e9, **kwargs}
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                serving.max_concurrent_requests(**args)

    def test_tpot_rejects_empty_batch(self) -> None:
        for batch in (0, -1):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                serving.tpot_lower_bound_seconds(
                    batch, 140e9, 70e9, 0, 989.4e12, 3.35e12
                )
        with self.assertRaises(ValueError):
            serving.tpot_lower_bound_seconds(8, 140e9, 70e9, -1.0, 989.4e12, 3.35e12)

    def test_throughput_rejects_non_positive_step(self) -> None:
        for args in ((8, 0), (8, -0.02), (-1, 0.02)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                serving.tokens_per_second(*args)

    def test_cost_rejects_invalid_inputs(self) -> None:
        for args in ((3.6, 0), (3.6, -1000), (-3.6, 1000)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                serving.cost_per_million_tokens(*args)


if __name__ == "__main__":
    unittest.main()
