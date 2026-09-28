import unittest

from infra_calc import cost

ANCHORS = ("references/source-book/chapter11.md:460",)


class CostTest(unittest.TestCase):
    def test_example_11_5_reasoning_length(self) -> None:
        # 10 000 входных; 200 видимых + 1000 или 100 токенов рассуждения; $2 и $10 за миллион
        self.assertAlmostEqual(cost.call_cost(10_000, 1_200, 2, 10), 0.032)
        self.assertAlmostEqual(cost.call_cost(10_000, 300, 2, 10), 0.023)

    def test_example_11_6_prefix_cache(self) -> None:
        # первый вызов создаёт кэш 8 000 токенов, девять следующих читают его
        first = cost.call_cost(
            2_000, 0, 2, 10, cache_write_tokens=8_000, cache_write_price=2.5
        )
        later = cost.call_cost(
            2_000, 0, 2, 10, cached_tokens=8_000, cache_read_price=0.2
        )
        self.assertAlmostEqual(first + 9 * later, 0.0744)

    def test_cost_per_accepted_task(self) -> None:
        self.assertEqual(cost.cost_per_accepted_task(0.03, 0.6), 0.05)

    def test_call_cost_rejects_negative_inputs(self) -> None:
        names = (
            "input_tokens",
            "output_tokens",
            "input_price",
            "output_price",
            "cached_tokens",
            "cache_read_price",
            "cache_write_tokens",
            "cache_write_price",
        )
        base = dict.fromkeys(names, 1)
        for name in names:
            with self.subTest(name=name), self.assertRaises(ValueError):
                cost.call_cost(**{**base, name: -1})

    def test_cost_per_accepted_task_rejects_invalid_inputs(self) -> None:
        # вероятность успеха вне (0, 1] дала бы бесконечную или заниженную цену
        for args in ((0.03, 0), (0.03, -0.5), (0.03, 1.5), (-0.03, 0.6)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                cost.cost_per_accepted_task(*args)


if __name__ == "__main__":
    unittest.main()
