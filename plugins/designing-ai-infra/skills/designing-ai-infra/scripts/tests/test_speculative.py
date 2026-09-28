import unittest

from infra_calc import speculative

ANCHORS = ("references/source-book/chapter8.md:536",)


class SpeculativeTest(unittest.TestCase):
    def test_expected_tokens(self) -> None:
        # chapter8.md:536: E[N] = 1 + a + a² + a³ + a⁴; 1.33 при a=1/4, 3.05 при a=3/4
        self.assertEqual(round(speculative.expected_tokens_per_round(0.25, 4), 2), 1.33)
        self.assertEqual(round(speculative.expected_tokens_per_round(0.75, 4), 2), 3.05)

    def test_mean_time_is_weighted_by_tokens(self) -> None:
        # два раунда по 1.5 мс, 1 и 5 токенов: 0.5 мс на токен, а не 0.9
        self.assertEqual(
            speculative.mean_time_per_token([1.5e-3, 1.5e-3], [1, 5]), 0.5e-3
        )

    def test_breakeven(self) -> None:
        # полное время раунда должно быть меньше E[N] × 26.26 мс
        self.assertAlmostEqual(
            speculative.breakeven_round_seconds(3.05, 26.26e-3), 80.093e-3, places=6
        )

    def test_expected_tokens_rejects_invalid_inputs(self) -> None:
        # доля принятия вне [0, 1] и отрицательная длина черновика дают бессмысленный E[N]
        for args in ((-0.1, 4), (1.1, 4), (0.5, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                speculative.expected_tokens_per_round(*args)

    def test_mean_time_rejects_invalid_rounds(self) -> None:
        # раунд всегда даёт хотя бы один проверенный токен целевой модели
        for seconds, tokens in (
            ([], []),
            ([1e-3], [1, 2]),
            ([1e-3], [0]),
            ([-1e-3], [1]),
        ):
            with (
                self.subTest(seconds=seconds, tokens=tokens),
                self.assertRaises(ValueError),
            ):
                speculative.mean_time_per_token(seconds, tokens)

    def test_breakeven_rejects_invalid_inputs(self) -> None:
        # E[N] не бывает меньше 1: один токен раунд даёт всегда
        for args in ((0.5, 26.26e-3), (3.05, 0), (3.05, -1e-3)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                speculative.breakeven_round_seconds(*args)


if __name__ == "__main__":
    unittest.main()
