import math
import unittest

from infra_calc import cost

ANCHORS = (
    "references/source-book/chapter11.md:460",
    "references/source-book/chapter11.md:488",
    "references/source-book/chapter11.md:515",
)


class CostTest(unittest.TestCase):
    def test_example_11_5_reasoning_length(self) -> None:
        # chapter11.md:471,475: «C_{1000}=\frac{10000\times2+(1000+200)\times10}{10^6}=0.032»,
        # «C_{100}=\frac{10000\times2+(100+200)\times10}{10^6}=0.023»
        self.assertAlmostEqual(cost.call_cost(10_000, 1_200, 2, 10), 0.032)
        self.assertAlmostEqual(cost.call_cost(10_000, 300, 2, 10), 0.023)

    def test_example_11_6_prefix_cache(self) -> None:
        # chapter11.md:504: «Без кэша стоимость входных данных за десять раундов составляет
        # 0,20 доллара»; 8 000 токенов префикса и 2 000 новых — непересекающиеся категории
        self.assertAlmostEqual(10 * cost.call_cost(10_000, 0, 2, 10), 0.20)
        # chapter11.md:507-508: первый вызов создаёт кэш 8 000 токенов, девять следующих читают его, «=0.0744»
        first = cost.call_cost(
            2_000, 0, 2, 10, cache_write_tokens=8_000, cache_write_price=2.5
        )
        later = cost.call_cost(
            2_000, 0, 2, 10, cached_tokens=8_000, cache_read_price=0.2
        )
        self.assertAlmostEqual(first + 9 * later, 0.0744)

    def test_example_11_7_cost_per_accepted_task(self) -> None:
        # chapter11.md:521-530: 19 000 токенов префикса из кэша и 1 000 новых; A: 1 / 0.1 / 5 $,
        # 1 800 + 200 выходных, успех 0.80; B: 2 / 0.2 / 10 $, 100 + 200 выходных.
        # chapter11.md:532: «стоимость каждого вызова составляет 0,0129 доллара, а средняя стоимость
        # успешно выполненной задачи — $0.0129/0.8\approx0.0161$ доллара. При попадании в кэш стоимость
        # каждого вызова B составляет 0,0088 доллара, а без попадания — 0,0430 доллара.»
        call_a = cost.call_cost(
            1_000, 2_000, 1, 5, cached_tokens=19_000, cache_read_price=0.1
        )
        self.assertAlmostEqual(call_a, 0.0129)
        self.assertEqual(round(cost.cost_per_accepted_task(call_a, 0.8), 4), 0.0161)
        self.assertAlmostEqual(
            cost.call_cost(
                1_000, 300, 2, 10, cached_tokens=19_000, cache_read_price=0.2
            ),
            0.0088,
        )
        self.assertAlmostEqual(cost.call_cost(20_000, 300, 2, 10), 0.0430)

    def test_call_cost_rejects_invalid_inputs(self) -> None:
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
            for value in (-1, math.nan, math.inf):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    cost.call_cost(**{**base, name: value})

    def test_expected_tokens_at_partial_cache_hit_rate(self) -> None:
        # chapter11.md:532-538: доля попаданий B в кэш h; при промахе 19 000 токенов
        # префикса оплачиваются как обычный вход, поэтому ожидаемые числа токенов
        # вызова дробные в общем случае. «Приравняв среднюю стоимость B к средней
        # стоимости A, получим $h\approx79.5\%$» (A — 0.0161); «Если доля попаданий
        # снижается до 50%, стоимость успешной задачи B возрастает примерно до 0,0264»;
        # chapter11.md:530: вероятность успеха B 0.98 (формула C_B(h) делит на 0.98)
        def expected_b(hit_rate: float) -> float:
            call = cost.call_cost(
                1_000 + 19_000 * (1 - hit_rate),
                300,
                2,
                10,
                cached_tokens=19_000 * hit_rate,
                cache_read_price=0.2,
            )
            return cost.cost_per_accepted_task(call, 0.98)

        self.assertEqual(round(expected_b(0.795), 4), 0.0161)
        self.assertEqual(round(expected_b(0.5), 4), 0.0264)
        self.assertAlmostEqual(cost.call_cost(0.5, 0.25, 2, 10), 3.5e-6)

    def test_call_cost_rejects_boolean_tokens(self) -> None:
        # True — не среднее число токенов, а перепутанный аргумент
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "cache_write_tokens",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                cost.call_cost(
                    **{
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "input_price": 1,
                        "output_price": 1,
                        name: True,
                    }
                )

    def test_cost_per_accepted_task_rejects_invalid_inputs(self) -> None:
        # вероятность успеха вне (0, 1] дала бы бесконечную или заниженную цену
        for args in (
            (0.03, 0),
            (0.03, -0.5),
            (0.03, 1.5),
            (0.03, math.nan),
            (-0.03, 0.6),
            (math.inf, 0.6),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                cost.cost_per_accepted_task(*args)


if __name__ == "__main__":
    unittest.main()
