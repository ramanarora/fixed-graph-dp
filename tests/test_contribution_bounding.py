"""Small-instance correctness checks for the public contribution selector."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np
from multiattr import greedy_cb, incidence, lp_rounding_cb, lp_weighted_greedy, solve_lp


class ContributionBoundingTests(unittest.TestCase):
    def assert_feasible_maximal(self, edges, n_vertices, k, selected):
        self.assertEqual(len(selected), len(set(selected)))
        degrees = np.zeros(n_vertices, dtype=int)
        for j in selected:
            degrees[edges[j]] += 1
        self.assertTrue(np.all(degrees <= k))
        for j, edge in enumerate(edges):
            if j not in selected:
                self.assertTrue(any(degrees[v] == k for v in edge))

    def test_empty_and_already_feasible(self):
        rng = np.random.default_rng(1)
        self.assertEqual(lp_rounding_cb([], 0, 1, rng), [])
        with patch("multiattr.solve_lp", side_effect=AssertionError("LP unnecessary")):
            self.assertEqual(lp_rounding_cb([[0], [0], [1]], 2, 2, rng), [0, 1, 2])

    def test_singleton_safeguard_is_automatic(self):
        selected = lp_rounding_cb([[0], [0]], 1, 1, np.random.default_rng(2))
        self.assertEqual(len(selected), 1)

    def test_integral_returned_lp_solution_is_exact(self):
        edges = [[0], [1], [0, 1]]
        for seed in range(20):
            selected = lp_weighted_greedy(edges, 2, 1, [1, 1, 0], np.random.default_rng(seed))
            self.assertEqual(set(selected), {0, 1})

    def test_zero_weight_tail_completes_selection(self):
        edges = [[0], [1], [0, 1]]
        selected = lp_weighted_greedy(edges, 2, 1, [1, 0, 0], np.random.default_rng(3))
        self.assertEqual(selected, [0, 1])
        self.assert_feasible_maximal(edges, 2, 1, selected)

    def test_identical_attribution_sets_are_distinct_examples(self):
        edges = [[0], [0], [0], [1]]
        selected = lp_rounding_cb(edges, 2, 2, np.random.default_rng(4))
        self.assertEqual(len(selected), 3)
        self.assertEqual(sum(j < 3 for j in selected), 2)

    def test_feasibility_maximality_and_deterministic_r_bound(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            edges = [sorted(rng.choice(5, size=int(rng.integers(1, 4)), replace=False).tolist())
                     for _ in range(9)]
            k = int(rng.integers(1, 3))
            y, _ = solve_lp(edges, 5, k)
            A = incidence(edges, 5)
            opt = max(sum(bits) for bits in itertools.product((0, 1), repeat=len(edges))
                      if np.all(A @ bits <= k))
            for _ in range(4):
                selected = lp_weighted_greedy(edges, 5, k, y, rng)
                self.assert_feasible_maximal(edges, 5, k, selected)
                self.assertLessEqual(opt, max(map(len, edges)) * len(selected))

    def test_first_appearance_coupling_for_fixed_stream(self):
        edges = [[0, 1], [1, 2], [2, 3], [0], [3]]
        stream = [0, 1, 1, 0, 3, 2, 4, 2, 1, 4]
        selected, degrees = [], np.zeros(4, dtype=int)
        for j in stream:
            if j not in selected and all(degrees[v] < 1 for v in edges[j]):
                selected.append(j)
                degrees[edges[j]] += 1
        first_appearances = list(dict.fromkeys(stream))
        self.assertEqual(selected, greedy_cb(edges, 4, 1, first_appearances))

    def test_one_lp_solve_and_reproducibility(self):
        edges = [[0], [0], [1], [0, 1]]
        with patch("multiattr.solve_lp", wraps=solve_lp) as solver:
            a = lp_rounding_cb(edges, 2, 1, np.random.default_rng(6), repeats=8)
            self.assertEqual(solver.call_count, 1)
        b = lp_rounding_cb(edges, 2, 1, np.random.default_rng(6), repeats=8)
        self.assertEqual(a, b)

    def test_invalid_budget_or_repetition_count(self):
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                lp_rounding_cb([[0]], 1, bad, np.random.default_rng(7))
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                lp_rounding_cb([[0]], 1, 1, np.random.default_rng(7), repeats=bad)


if __name__ == "__main__":
    unittest.main()
