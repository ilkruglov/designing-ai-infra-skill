import math
import unittest

from infra_calc import checks

# Проверки входов — не калькулятор книги: чисел книги здесь нет, якорей тоже.
ANCHORS: tuple[str, ...] = ()


class RequirePositiveTest(unittest.TestCase):
    def test_accepts_positive(self) -> None:
        checks.require_positive("peak", 1e-9)

    def test_rejects_zero_and_negative(self) -> None:
        for value in (0, -1.0):
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                checks.require_positive("peak", value)
            self.assertIn("peak должен быть больше нуля", str(caught.exception))

    def test_nan_is_reported_as_not_a_number(self) -> None:
        # «должен быть больше нуля: nan» указывало бы не на ту ошибку
        with self.assertRaises(ValueError) as caught:
            checks.require_positive("peak", math.nan)
        self.assertIn("peak должен быть конечным числом", str(caught.exception))


class RequireNonNegativeTest(unittest.TestCase):
    def test_accepts_zero(self) -> None:
        checks.require_non_negative("flops", 0)

    def test_rejects_negative(self) -> None:
        with self.assertRaises(ValueError) as caught:
            checks.require_non_negative("flops", -1e-9)
        self.assertIn("flops не может быть отрицательным", str(caught.exception))

    def test_nan_is_reported_as_not_a_number(self) -> None:
        # «не может быть отрицательным: nan» указывало бы не на ту ошибку
        with self.assertRaises(ValueError) as caught:
            checks.require_non_negative("flops", math.nan)
        self.assertIn("flops должен быть конечным числом", str(caught.exception))


class RequireFiniteTest(unittest.TestCase):
    def test_rejects_infinity(self) -> None:
        # бесконечная полоса или пик дают нулевое время — правдоподобный, но ложный ответ
        for check in (checks.require_positive, checks.require_non_negative):
            with (
                self.subTest(check=check.__name__),
                self.assertRaises(ValueError) as caught,
            ):
                check("peak", math.inf)
            self.assertIn("peak должен быть конечным числом", str(caught.exception))

    def test_require_finite_rejects_nan_and_infinity(self) -> None:
        checks.require_finite("expected_tokens", -1.5)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                checks.require_finite("expected_tokens", value)
            self.assertIn(
                "expected_tokens должен быть конечным числом", str(caught.exception)
            )


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


class GrammaticalFormTest(unittest.TestCase):
    """Сказуемое согласуется с подлежащим: «интенсивность ... не может быть отрицательной»."""

    def test_feminine(self) -> None:
        with self.assertRaises(ValueError) as caught:
            checks.require_non_negative("интенсивность класса 'a'", -1, form="f")
        self.assertEqual(
            str(caught.exception),
            "интенсивность класса 'a' не может быть отрицательной: -1",
        )
        with self.assertRaises(ValueError) as caught:
            checks.require_positive("интенсивность", math.nan, form="f")
        self.assertIn(
            "интенсивность должна быть конечным числом", str(caught.exception)
        )

    def test_plural(self) -> None:
        cases = (
            ((2.5, 0), "входные токены должны быть целыми числами: 2.5"),
            ((-1, 0), "входные токены не могут быть отрицательными: -1"),
            ((0, 1), "входные токены должны быть не меньше 1: 0"),
        )
        for (value, minimum), message in cases:
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                checks.require_int_at_least("входные токены", value, minimum, form="pl")
            self.assertEqual(str(caught.exception), message)

    def test_neuter(self) -> None:
        with self.assertRaises(ValueError) as caught:
            checks.require_non_negative("время раунда", math.inf, form="n")
        self.assertIn("время раунда должно быть конечным числом", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            checks.require_positive("время раунда", 0, form="n")
        self.assertIn("время раунда должно быть больше нуля", str(caught.exception))

    def test_unknown_form(self) -> None:
        with self.assertRaises(ValueError):
            checks.require_positive("x", 1, form="dual")


if __name__ == "__main__":
    unittest.main()
