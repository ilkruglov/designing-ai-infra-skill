import math
import unittest

from infra_calc import collectives

ANCHORS = (
    "references/source-book/chapter6.md:490",
    "calculations/results/all-to-all-qwen235-t64-balanced.json#sha256=33ad74d2ac4b1441f7b24750e9ed450116e613463e095836147c3478b6c40743",
    "calculations/results/all-to-all-qwen235-t64-hotspot.json#sha256=4fdd302a8ddb0083b582c54f945aeb9d0e5849b9626e370d7fcf7c6fbe7a1fd3",
    "calculations/results/tree-qwen3-8b-t1-p5.json#sha256=00b2c04343387230c21c26b5faa6bc8f90234c227269957f328d0dfbf682d667",
    "calculations/results/tree-qwen3-32b-t1-p8-h100.json#sha256=310710bf34b7d88a7ac7bd35335247b12c1b627549fd067ab52ed37691f07448",
    "calculations/results/tree-qwen3-32b-t8192-p8-h100.json#sha256=eed24bed5ea4d6ca234b9477a15d83b628c37c9e961a3600e7e2a8601d32f64f",
)


class RingTest(unittest.TestCase):
    def test_tp8_example(self) -> None:
        # chapter6.md:518, формула (6-9): M = 10 KiB, B = 450 GB/s, α = 0.822 μs;
        # «каждая карта отправляет 17,5 KiB, ... а одна операция — около 11,55 μs»
        seconds = collectives.ring_allreduce_seconds(8, 10 * 1024, 450e9, 0.822e-6)
        self.assertEqual(round(seconds * 1e6, 2), 11.55)
        self.assertEqual(
            collectives.ring_bytes_sent_per_device(8, 10 * 1024) / 1024, 17.5
        )

    def test_tp_step_table(self) -> None:
        # chapter6.md:520-525, таблица «Число карт TP | Локальный доступ к памяти | 128 редукций |
        # Время шага»: 29,35 / 14,89 / 7,97 / 5,15 ms
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
        # all-to-all-qwen235-t64-balanced.json: 8 рангов по 64 назначения, 8192 байт на вектор;
        # summary: dispatch_network_send_bytes, dispatch_endpoint_service_lower_seconds,
        # dispatch_pairwise_modeled_seconds
        counts = [[64] * 8 for _ in range(8)]
        network, endpoint, pairwise = collectives.all_to_all_phase(
            counts, 8192, 50e9, 2e-6
        )
        self.assertEqual(network, 29_360_128)
        self.assertAlmostEqual(endpoint, 7.340032e-05, places=12)
        self.assertAlmostEqual(pairwise, 8.740032e-05, places=12)

    def test_all_to_all_hotspot(self) -> None:
        # all-to-all-qwen235-t64-hotspot.json: все 512 назначений каждого ранга идут в ранг 0;
        # summary: те же dispatch_network_send_bytes, но 5.8720256e-4 и 6.0120256e-4 s
        counts = [[512] + [0] * 7 for _ in range(8)]
        network, endpoint, pairwise = collectives.all_to_all_phase(
            counts, 8192, 50e9, 2e-6
        )
        self.assertEqual(network, 29_360_128)
        self.assertAlmostEqual(endpoint, 5.8720256e-4, places=12)
        self.assertAlmostEqual(pairwise, 6.0120256e-4, places=12)

    def test_tree_rounds(self) -> None:
        # chapter6.md:535: «Четыре карты сначала выполняют попарную редукцию ... для редукции
        # требуется два раунда; широковещательная рассылка занимает ещё два раунда, итого четыре.»
        self.assertEqual(collectives.tree_allreduce_rounds(4), 4)

    def test_tree_rounds_not_power_of_two(self) -> None:
        # tree-qwen3-8b-t1-p5.json: summary.all_reduce_rounds = 6 для пяти участников,
        # то есть 2·ceil(log2 5); формула (6-10) книги записана только для степеней двойки
        self.assertEqual(collectives.tree_allreduce_rounds(5), 6)

    def test_tree_time_book_and_author(self) -> None:
        # chapter6.md:541, формула (6-10): «Для восьми карт и 10 KiB время составляет около
        # 5,07 μs»; «80 MiB ... кольцевому алгоритму потребуется около 0,34 ms, а древовидному —
        # около 1,12 ms»; tree-qwen3-32b-t1-p8-h100.json и tree-qwen3-32b-t8192-p8-h100.json:
        # all_reduce_modeled_seconds 5.068533333333334e-06 и 0.0011234130666666667
        small = collectives.tree_allreduce_seconds(8, 10 * 1024, 450e9, 0.822e-6)
        self.assertEqual(round(small * 1e6, 2), 5.07)
        self.assertAlmostEqual(small, 5.068533333333334e-06, places=15)
        large = collectives.tree_allreduce_seconds(8, 80 * 2**20, 450e9, 0.822e-6)
        self.assertEqual(round(large * 1e3, 2), 1.12)
        self.assertAlmostEqual(large, 0.0011234130666666667, places=15)
        ring = collectives.ring_allreduce_seconds(8, 80 * 2**20, 450e9, 0.822e-6)
        self.assertEqual(round(ring * 1e3, 2), 0.34)

    def test_tree_time_hand_derived(self) -> None:
        # вывод вручную: пять участников — 2·ceil(log2 5) = 6 раундов по α + M/B;
        # один участник — ноль раундов
        self.assertAlmostEqual(
            collectives.tree_allreduce_seconds(5, 1e6, 1e9, 1e-5), 6 * (1e-5 + 1e-3)
        )
        self.assertEqual(collectives.tree_allreduce_seconds(1, 1e6, 1e9, 1e-5), 0)

    def test_crossover_book(self) -> None:
        # chapter6.md:541: «Если приравнять две формулы, точка пересечения будет находиться
        # примерно на 680 KiB» (8 карт, 0,822 μs, 450 GB/s)
        crossover = collectives.ring_tree_crossover_bytes(8, 450e9, 0.822e-6)
        self.assertEqual(round(crossover / 1024, -1), 680)
        # выше точки кольцо быстрее, ниже — дерево
        for message, tree_faster in ((crossover / 2, True), (crossover * 2, False)):
            tree = collectives.tree_allreduce_seconds(8, message, 450e9, 0.822e-6)
            ring = collectives.ring_allreduce_seconds(8, message, 450e9, 0.822e-6)
            self.assertEqual(tree < ring, tree_faster)

    def test_crossover_hand_derived(self) -> None:
        # вывод вручную: n = 4, L = 2: M_* = αB(L − (n − 1)) / ((n − 1)/n − L) = 0.8·αB;
        # при n ≤ 3 дерево не быстрее кольца ни при каком M > 0 — точка 0;
        # при n = 1 обе операции пусты — точки нет
        self.assertAlmostEqual(
            collectives.ring_tree_crossover_bytes(4, 1e9, 1e-5), 0.8 * 1e4
        )
        self.assertEqual(collectives.ring_tree_crossover_bytes(3, 1e9, 1e-5), 0)
        self.assertEqual(collectives.ring_tree_crossover_bytes(2, 1e9, 1e-5), 0)
        self.assertIsNone(collectives.ring_tree_crossover_bytes(1, 1e9, 1e-5))

    def test_tree_rejects_invalid_inputs(self) -> None:
        for args in (
            (0, 1e6, 1e9, 1e-5),
            (8, -1, 1e9, 1e-5),
            (8, 1e6, 0, 1e-5),
            (8, 1e6, 1e9, -1e-5),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                collectives.tree_allreduce_seconds(*args)
        with self.assertRaises(ValueError):
            collectives.ring_tree_crossover_bytes(8, 0, 1e-5)

    def test_ring_rejects_invalid_inputs(self) -> None:
        # ноль или дробь карт — деление на ноль или ложное число раундов, нулевая полоса — бесконечное время
        for args in (
            (0, 10 * 1024, 450e9, 0.822e-6),
            (8.0, 10 * 1024, 450e9, 0.822e-6),
            (8, -1, 450e9, 0.822e-6),
            (8, 10 * 1024, 0, 0.822e-6),
            (8, 10 * 1024, math.inf, 0.822e-6),
            (8, 10 * 1024, 450e9, -0.822e-6),
            (8, 10 * 1024, 450e9, math.nan),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                collectives.ring_allreduce_seconds(*args)

    def test_ring_bytes_rejects_nan_and_fractional_devices(self) -> None:
        # раньше ring_bytes_sent_per_device(nan, 1e6) → nan, (2.5, 1e6) → 1.2e6
        for args in (
            (math.nan, 1e6),
            (2.5, 1e6),
            (0, 10 * 1024),
            (True, 10 * 1024),
            (8, -1),
            (8, math.inf),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                collectives.ring_bytes_sent_per_device(*args)

    def test_tree_rejects_invalid_devices(self) -> None:
        for devices in (0, -4, 4.0, math.nan):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                collectives.tree_allreduce_rounds(devices)

    def test_tp_step_rejects_invalid_inputs(self) -> None:
        for args in (
            (29.35e-3, 0, 128, 10 * 1024, 450e9, 0.822e-6),
            (29.35e-3, 2.0, 128, 10 * 1024, 450e9, 0.822e-6),
            (29.35e-3, 8, -1, 10 * 1024, 450e9, 0.822e-6),
            (29.35e-3, 8, 1.5, 10 * 1024, 450e9, 0.822e-6),
            (-29.35e-3, 8, 128, 10 * 1024, 450e9, 0.822e-6),
            (math.nan, 8, 128, 10 * 1024, 450e9, 0.822e-6),
            (29.35e-3, 1, 128, 10 * 1024, 0, 0.822e-6),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                collectives.tp_step_seconds(*args)

    def test_all_to_all_rejects_invalid_counts(self) -> None:
        # матрица назначений квадратная, из целых неотрицательных счётчиков
        for counts in (
            [],
            [[64, 64], [64]],
            [[64, -1], [64, 64]],
            [[64, 1.5], [64, 64]],
            [[64, math.nan], [64, 64]],
        ):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                collectives.all_to_all_phase(counts, 8192, 50e9, 2e-6)

    def test_all_to_all_rejects_invalid_link(self) -> None:
        counts = [[64] * 2 for _ in range(2)]
        for args in (
            (-1, 50e9, 2e-6),
            (8192.5, 50e9, 2e-6),
            (8192, 0, 2e-6),
            (8192, math.inf, 2e-6),
            (8192, 50e9, -2e-6),
            (8192, 50e9, math.nan),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                collectives.all_to_all_phase(counts, *args)


if __name__ == "__main__":
    unittest.main()
