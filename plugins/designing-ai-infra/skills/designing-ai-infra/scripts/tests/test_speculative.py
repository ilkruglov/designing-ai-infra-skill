import math
import unittest

from infra_calc import speculative

ANCHORS = ("references/source-book/chapter8.md:536",)


class SpeculativeTest(unittest.TestCase):
    def test_expected_tokens(self) -> None:
        # chapter8.md:552: «Для черновика AAAA значение $a=1/4$, а среднее число
        # выходных токенов за раунд составляет около 1,33. Для BBBB значение $a=3/4$,
        # а среднее число выходных токенов — около 3,05.»
        self.assertEqual(round(speculative.expected_tokens_per_round(0.25, 4), 2), 1.33)
        self.assertEqual(round(speculative.expected_tokens_per_round(0.75, 4), 2), 3.05)

    def test_mean_time_is_weighted_by_tokens(self) -> None:
        # chapter8.md:544: «два раунда длительностью по 1,5 мс выводят соответственно
        # 1 и 5 токенов. Суммарно получаются 3 мс и 6 токенов, то есть в среднем
        # 0,5 мс на токен.» Равновесное среднее по раундам дало бы 0,9 мс.
        self.assertEqual(
            speculative.mean_time_per_token([1.5e-3, 1.5e-3], [1, 5]), 0.5e-3
        )

    def test_breakeven_leaves_draft_budget(self) -> None:
        # chapter8.md:554: «После вычета 26,26 мс, необходимых для проверки, на запрос
        # черновика остаётся около 8,7 мс для AAAA и около 53,9 мс для BBBB.»
        # E[N] берётся точным (1.33203125 и 3.05078125), а не округлённым.
        verify = 26.26e-3
        budgets = [
            round(
                (
                    speculative.breakeven_round_seconds(
                        speculative.expected_tokens_per_round(acceptance, 4), verify
                    )
                    - verify
                )
                * 1e3,
                1,
            )
            for acceptance in (0.25, 0.75)
        ]
        self.assertEqual(budgets, [8.7, 53.9])

    def test_expected_tokens_rejects_invalid_inputs(self) -> None:
        # доля принятия вне [0, 1] и отрицательная или дробная длина черновика дают бессмысленный E[N]
        for args in (
            (-0.1, 4),
            (1.1, 4),
            (math.nan, 4),
            (0.5, -1),
            (0.5, 2.5),
            (0.5, 4.0),
            (0.5, True),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                speculative.expected_tokens_per_round(*args)

    def test_mean_time_rejects_invalid_rounds(self) -> None:
        # раунд всегда даёт целое число токенов, не меньше одного проверенного токена целевой модели
        for seconds, tokens in (
            ([], []),
            ([1e-3], [1, 2]),
            ([1e-3], [0]),
            ([1e-3], [1.5]),
            ([-1e-3], [1]),
            ([math.nan], [1]),
            ([math.inf], [1]),
        ):
            with (
                self.subTest(seconds=seconds, tokens=tokens),
                self.assertRaises(ValueError),
            ):
                speculative.mean_time_per_token(seconds, tokens)

    def test_breakeven_rejects_invalid_inputs(self) -> None:
        # E[N] не бывает меньше 1: один токен раунд даёт всегда; NaN и бесконечность — не число токенов
        for args in (
            (0.5, 26.26e-3),
            (math.nan, 0.01),
            (math.inf, 0.01),
            (3.05, 0),
            (3.05, -1e-3),
            (3.05, math.nan),
            (3.05, math.inf),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                speculative.breakeven_round_seconds(*args)

    def test_breakeven_names_non_finite_expected_tokens(self) -> None:
        # NaN и бесконечность — не «меньше 1», а не число: сообщение о конечности
        for value in (math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                speculative.breakeven_round_seconds(value, 0.01)
            self.assertIn("конечным числом", str(caught.exception))
            self.assertNotIn("меньше 1", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
