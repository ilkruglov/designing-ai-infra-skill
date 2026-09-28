import unittest

from infra_calc import collectives

ANCHORS = (
    "references/source-book/chapter6.md:473",
    "calculations/results/all-to-all-qwen235-t64-balanced.json#sha256=33ad74d2ac4b1441f7b24750e9ed450116e613463e095836147c3478b6c40743",
)


class RingTest(unittest.TestCase):
    def test_tp8_example(self) -> None:
        # chapter6.md:473, формула (6-9): M = 10 KiB, B = 450 GB/s, α = 0.822 μs
        seconds = collectives.ring_allreduce_seconds(8, 10 * 1024, 450e9, 0.822e-6)
        self.assertEqual(round(seconds * 1e6, 2), 11.55)
        self.assertEqual(
            collectives.ring_bytes_sent_per_device(8, 10 * 1024) / 1024, 17.5
        )

    def test_tp_step_table(self) -> None:
        # chapter6.md:473: локальный доступ 29.35 ms на TP1, 128 редукций на шаг
        steps = [
            round(
                collectives.tp_step_seconds(
                    29.35e-3, tp, 128, 10 * 1024, 450e9, 0.822e-6
                )
                * 1e3,
                2,
            )
            for tp in (1, 2, 4, 8)
        ]
        self.assertEqual(steps, [29.35, 14.89, 7.97, 5.15])

    def test_all_to_all_balanced(self) -> None:
        # all-to-all-qwen235-t64-balanced.json: 8 рангов по 64 назначения, 8192 байт на вектор
        counts = [[64] * 8 for _ in range(8)]
        network, endpoint, pairwise = collectives.all_to_all_phase(
            counts, 8192, 50e9, 2e-6
        )
        self.assertEqual(network, 29_360_128)
        self.assertAlmostEqual(endpoint, 7.340032e-05, places=12)
        self.assertAlmostEqual(pairwise, 8.740032e-05, places=12)

    def test_tree_rounds(self) -> None:
        # четыре карты: два раунда редукции и два рассылки
        self.assertEqual(collectives.tree_allreduce_rounds(4), 4)

    def test_ring_rejects_invalid_inputs(self) -> None:
        # ноль карт — деление на ноль, нулевая полоса — бесконечное время
        for args in (
            (0, 10 * 1024, 450e9, 0.822e-6),
            (8, -1, 450e9, 0.822e-6),
            (8, 10 * 1024, 0, 0.822e-6),
            (8, 10 * 1024, 450e9, -0.822e-6),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                collectives.ring_allreduce_seconds(*args)
        for args in ((0, 10 * 1024), (8, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                collectives.ring_bytes_sent_per_device(*args)

    def test_tree_rejects_no_devices(self) -> None:
        for devices in (0, -4):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                collectives.tree_allreduce_rounds(devices)

    def test_tp_step_rejects_invalid_inputs(self) -> None:
        for args in (
            (29.35e-3, 0, 128, 10 * 1024, 450e9, 0.822e-6),
            (29.35e-3, 8, -1, 10 * 1024, 450e9, 0.822e-6),
            (-29.35e-3, 8, 128, 10 * 1024, 450e9, 0.822e-6),
            (29.35e-3, 1, 128, 10 * 1024, 0, 0.822e-6),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                collectives.tp_step_seconds(*args)

    def test_all_to_all_rejects_invalid_counts(self) -> None:
        # матрица назначений квадратная, без отрицательных счётчиков
        for counts in ([], [[64, 64], [64]], [[64, -1], [64, 64]]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                collectives.all_to_all_phase(counts, 8192, 50e9, 2e-6)

    def test_all_to_all_rejects_invalid_link(self) -> None:
        counts = [[64] * 2 for _ in range(2)]
        for args in ((-1, 50e9, 2e-6), (8192, 0, 2e-6), (8192, 50e9, -2e-6)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                collectives.all_to_all_phase(counts, *args)


if __name__ == "__main__":
    unittest.main()
