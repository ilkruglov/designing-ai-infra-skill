import unittest

from infra_calc import queueing

ANCHORS = ("references/source-book/chapter3.md:75",)
CLASSES = {"long_input": (8192, 256), "long_output": (1024, 2048)}


class QueueingTest(unittest.TestCase):
    def test_example_3_1_uniform_mix(self) -> None:
        # chapter3.md:75, пример 3-1: 18 432 входных токенов/с и 4 604 шагов decode/с
        self.assertEqual(
            queueing.demand({"long_input": 2, "long_output": 2}, CLASSES),
            (18_432, 4_604),
        )

    def test_example_3_1_first_minute(self) -> None:
        # 29 900.8 и 1 736.8 в первую минуту изменяющейся смеси
        tokens, steps = queueing.demand(
            {"long_input": 3.6, "long_output": 0.4}, CLASSES
        )
        self.assertAlmostEqual(tokens, 29_900.8)
        self.assertAlmostEqual(steps, 1_736.8)

    def test_utilization_and_little(self) -> None:
        self.assertEqual(queueing.utilization(900, 1000), 0.9)
        self.assertEqual(queueing.littles_law_in_system(4, 2.5), 10)

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

    def test_utilization_rejects_non_positive_capacity(self) -> None:
        for args in ((900, 0), (900, -1000), (-900, 1000)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                queueing.utilization(*args)

    def test_little_rejects_negative_inputs(self) -> None:
        for args in ((-4, 2.5), (4, -2.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                queueing.littles_law_in_system(*args)


if __name__ == "__main__":
    unittest.main()
