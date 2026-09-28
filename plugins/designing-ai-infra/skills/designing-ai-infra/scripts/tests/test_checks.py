import math
import unittest

from infra_calc import checks

# Проверки входов — не калькулятор книги: чисел книги здесь нет, якорей тоже.
ANCHORS: tuple[str, ...] = ()


class RequirePositiveTest(unittest.TestCase):
    def test_accepts_positive(self) -> None:
        checks.require_positive("peak", 1e-9)

    def test_rejects_zero_negative_and_nan(self) -> None:
        for value in (0, -1.0, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                checks.require_positive("peak", value)
            self.assertIn("peak должен быть больше нуля", str(caught.exception))


class RequireNonNegativeTest(unittest.TestCase):
    def test_accepts_zero(self) -> None:
        checks.require_non_negative("flops", 0)

    def test_rejects_negative_and_nan(self) -> None:
        for value in (-1e-9, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                checks.require_non_negative("flops", value)
            self.assertIn("flops не может быть отрицательным", str(caught.exception))


class RequireIntAtLeastTest(unittest.TestCase):
    def test_accepts_integer_at_minimum(self) -> None:
        checks.require_int_at_least("batch", 1, 1)
        checks.require_int_at_least("history", 0, 0)

    def test_rejects_non_integers(self) -> None:
        for value in (2.5, 2.0, True, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                checks.require_int_at_least("batch", value, 1)
            self.assertIn("batch должен быть целым числом", str(caught.exception))

    def test_below_minimum_keeps_earlier_wording(self) -> None:
        with self.assertRaises(ValueError) as caught:
            checks.require_int_at_least("batch", 0, 1)
        self.assertEqual(str(caught.exception), "batch должен быть не меньше 1: 0")
        with self.assertRaises(ValueError) as caught:
            checks.require_int_at_least("history", -1, 0)
        self.assertEqual(
            str(caught.exception), "history не может быть отрицательным: -1"
        )


if __name__ == "__main__":
    unittest.main()
