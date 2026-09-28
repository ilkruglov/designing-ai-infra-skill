import math
import unittest

from infra_calc import queueing

ANCHORS = (
    "references/source-book/chapter3.md:75",
    "references/source-book/chapter8.md:598",
    "references/source-book/chapter11.md:62",
)
CLASSES = {"long_input": (8192, 256), "long_output": (1024, 2048)}


class QueueingTest(unittest.TestCase):
    def test_example_3_1_uniform_mix(self) -> None:
        # chapter3.md:85: «в среднем четыре запроса в секунду», доли классов равны;
        # chapter3.md:89: «| Равномерная смесь | 18,432 | 4,604 |»
        self.assertEqual(
            queueing.demand({"long_input": 2, "long_output": 2}, CLASSES),
            (18_432, 4_604),
        )

    def test_example_3_1_changing_mix(self) -> None:
        # chapter3.md:85: «в первую минуту соотношение классов равно $9:1$, а во вторую — $1:9$»
        # chapter3.md:90: «| Изменяющаяся во времени смесь, первая минута | 29,900.8 | 1,736.8 |»
        tokens, steps = queueing.demand(
            {"long_input": 3.6, "long_output": 0.4}, CLASSES
        )
        self.assertAlmostEqual(tokens, 29_900.8)
        self.assertAlmostEqual(steps, 1_736.8)
        # chapter3.md:91: «| Изменяющаяся во времени смесь, вторая минута | 6,963.2 | 7,471.2 |»
        tokens, steps = queueing.demand(
            {"long_input": 0.4, "long_output": 3.6}, CLASSES
        )
        self.assertAlmostEqual(tokens, 6_963.2)
        self.assertAlmostEqual(steps, 7_471.2)

    def test_one_card_is_overloaded_by_the_uniform_mix(self) -> None:
        # chapter3.md:137: «одна карта может выполнять не более приблизительно 2,796 шага
        # decode в секунду, что ниже средней потребности равномерной смеси в 4,604 шага»
        self.assertGreater(queueing.utilization(4_604, 2_796), 1)

    def test_littles_law(self) -> None:
        # chapter11.md:76: «Каждая задача занимает среду на 30 секунд, а каждую секунду
        # использовать среду начинают 10 задач, поэтому ... в среднем 300 сред»
        self.assertEqual(queueing.littles_law_in_system(10, 30), 300)

    def test_demand_rejects_invalid_classes(self) -> None:
        # ноль выходных токенов дал бы −1 шаг decode на запрос
        for arrivals, classes in (
            ({"a": 1.0}, {"a": (1024, 0)}),
            ({"a": 1.0}, {"a": (-1, 256)}),
            ({"a": -1.0}, {"a": (1024, 256)}),
        ):
            with (
                self.subTest(arrivals=arrivals, classes=classes),
                self.assertRaises(ValueError),
            ):
                queueing.demand(arrivals, classes)

    def test_class_messages_agree_in_gender(self) -> None:
        cases = (
            (({"a": -1.0}, {"a": (1024, 256)}), "интенсивность класса 'a' не может быть отрицательной"),
            (({"a": 1.0}, {"a": (-1, 256)}), "входные токены класса 'a' не могут быть отрицательными"),
            (({"a": 1.0}, {"a": (1024, 0)}), "выходные токены класса 'a' должны быть не меньше 1"),
            (({"a": 1.0}, {"a": (1024.5, 256)}), "входные токены класса 'a' должны быть целыми числами"),
        )  # fmt: skip
        for args, message in cases:
            with self.subTest(message=message), self.assertRaises(ValueError) as caught:
                queueing.demand(*args)
            self.assertIn(message, str(caught.exception))

    def test_utilization_rejects_non_positive_capacity(self) -> None:
        for args in ((900, 0), (900, -1000), (-900, 1000)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                queueing.utilization(*args)

    def test_little_rejects_negative_inputs(self) -> None:
        for args in ((-4, 2.5), (4, -2.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                queueing.littles_law_in_system(*args)

    def test_unknown_class_names_known_ones(self) -> None:
        with self.assertRaises(ValueError) as caught:
            queueing.demand({"short": 1.0}, CLASSES)
        self.assertIn("long_input", str(caught.exception))

    def test_fractional_tokens_and_nan_rate_are_rejected(self) -> None:
        for arrivals, classes in (
            ({"a": 1.0}, {"a": (1024.5, 256)}),
            ({"a": 1.0}, {"a": (1024, 256.5)}),
            ({"a": math.nan}, {"a": (1024, 256)}),
        ):
            with (
                self.subTest(arrivals=arrivals, classes=classes),
                self.assertRaises(ValueError),
            ):
                queueing.demand(arrivals, classes)


if __name__ == "__main__":
    unittest.main()
