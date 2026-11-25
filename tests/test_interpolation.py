"""Run with: PYTHONPATH=src python -m unittest discover -s tests -v."""

from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from butterfly_motion.interpolation import interpolate_cubic


class InterpolateCubicTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(10, dtype=float)
        self.points = np.column_stack((t**3 - 2*t, 3*t**2 + t, -t**3 + 4))

    def test_recovers_cubic_with_single_and_consecutive_gaps(self):
        for gaps in ([4], [3, 4, 5]):
            with self.subTest(gaps=gaps):
                mask = np.ones(10, dtype=bool)
                mask[gaps] = False
                damaged = self.points.copy()
                damaged[~mask] = np.nan
                result = interpolate_cubic(damaged, mask)
                assert_allclose(result, self.points, atol=1e-10)
                assert_array_equal(result[mask], self.points[mask])

    def test_explicit_mask_hides_finite_values(self):
        mask = np.ones(10, dtype=bool)
        mask[4] = False
        damaged = self.points.copy()
        damaged[4] = 99999
        assert_allclose(interpolate_cubic(damaged, mask), self.points, atol=1e-10)

    def test_automatic_mask_excludes_whole_nonfinite_point(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                damaged = self.points.copy()
                damaged[4, 1] = value
                assert_allclose(interpolate_cubic(damaged), self.points, atol=1e-10)

    def test_boundary_gaps_remain_nan_including_finite_placeholders(self):
        mask = np.ones(10, dtype=bool)
        mask[[0, 4, 9]] = False
        result = interpolate_cubic(self.points, mask)
        self.assertTrue(np.isnan(result[[0, 9]]).all())
        assert_allclose(result[4], self.points[4], atol=1e-10)
        assert_array_equal(result[mask], self.points[mask])

    def test_automatic_mask_preserves_boundary_gaps(self):
        damaged = self.points.copy()
        damaged[[0, 9]] = np.nan
        result = interpolate_cubic(damaged)
        self.assertTrue(np.isnan(result[[0, 9]]).all())
        assert_array_equal(result[1:9], self.points[1:9])

    def test_complete_input_returns_independent_float_copy(self):
        points = self.points.astype(int)
        for mask in (None, np.ones(10, dtype=bool)):
            with self.subTest(mask=mask):
                result = interpolate_cubic(points, mask)
                assert_array_equal(result, points)
                self.assertEqual(result.dtype.kind, "f")
                self.assertFalse(np.shares_memory(result, points))

    def test_complete_single_frame_needs_no_interpolation(self):
        assert_array_equal(interpolate_cubic(self.points[:1]), self.points[:1])

    def test_two_observations_use_linear_interpolation(self):
        points = np.array([[0., 1., 2.], [np.nan]*3, [4., 5., 6.]])
        assert_allclose(interpolate_cubic(points)[1], [2, 3, 4])

    def test_three_observations_recover_quadratic(self):
        t = np.arange(5, dtype=float)
        points = np.column_stack((t*t, 2*t*t + t, -t*t))
        mask = np.array([True, False, True, False, True])
        assert_allclose(interpolate_cubic(points, mask), points, atol=1e-12)

    def test_insufficient_observations_raise(self):
        for mask in (np.zeros(10, dtype=bool), np.arange(10) == 4):
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(ValueError, "Not enough points"):
                    interpolate_cubic(self.points, mask)
        with self.assertRaisesRegex(ValueError, "Not enough points"):
            interpolate_cubic(np.full((5, 3), np.nan))

    def test_observed_nonfinite_coordinates_raise(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                damaged = self.points.copy()
                damaged[4, 0] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    interpolate_cubic(damaged, np.ones(10, dtype=bool))

    def test_invalid_trajectory_shape_or_dtype_raises(self):
        for points in (np.empty((0, 3)), np.zeros(3), np.zeros((4, 2)),
                       np.zeros((4, 3, 1)), np.array([["a"]*3]),
                       np.ones((4, 3), dtype=complex)):
            with self.subTest(shape=points.shape, dtype=points.dtype):
                with self.assertRaises(ValueError):
                    interpolate_cubic(points)

    def test_invalid_mask_shape_or_dtype_raises(self):
        for mask in (np.ones(9, dtype=bool), np.ones((10, 1), dtype=bool),
                     np.ones(10, dtype=int), np.ones(10, dtype=float), True):
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(ValueError, "mask"):
                    interpolate_cubic(self.points, mask)

    def test_inputs_are_not_modified(self):
        damaged = self.points.copy()
        damaged[4] = np.nan
        mask = np.isfinite(damaged).all(axis=1)
        before, mask_before = damaged.copy(), mask.copy()
        interpolate_cubic(damaged, mask)
        assert_array_equal(damaged, before)
        assert_array_equal(mask, mask_before)

    def test_repository_data_all_18_sequences_and_54_xyz_blocks(self):
        root = Path(__file__).resolve().parents[1] / "data" / "raw"
        paths = sorted(root.glob("*.txt"))
        self.assertEqual(len(paths), 18)
        gaps = 0
        for path in paths:
            data = np.loadtxt(path)
            self.assertEqual(data.shape[1], 9)
            for offset in (0, 3, 6):
                with self.subTest(file=path.name, xyz_block=offset // 3):
                    points = data[:, offset:offset + 3]
                    before = points.copy()
                    observed = np.isfinite(points).all(axis=1)
                    result = interpolate_cubic(points)
                    assert_array_equal(result[observed], points[observed])
                    assert_array_equal(points, before)
                    anchors = np.flatnonzero(observed)
                    interior = (~observed & (np.arange(len(points)) > anchors[0])
                                & (np.arange(len(points)) < anchors[-1]))
                    gaps += int((~observed).sum())
                    self.assertTrue(np.isfinite(result[interior]).all())
                    outside = ~observed & ~interior
                    self.assertTrue(np.isnan(result[outside]).all())
        self.assertGreater(gaps, 0)

    def test_repository_sequences_with_deliberately_hidden_interior_points(self):
        root = Path(__file__).resolve().parents[1] / "data" / "raw"
        for path in sorted(root.glob("*.txt")):
            data = np.loadtxt(path)
            for offset in (0, 3, 6):
                with self.subTest(file=path.name, xyz_block=offset // 3):
                    points = data[:, offset:offset + 3]
                    observed = np.isfinite(points).all(axis=1)
                    middle = len(points) // 2
                    self.assertTrue(observed[middle - 1:middle + 2].all())
                    observed[middle - 1:middle + 2] = False
                    result = interpolate_cubic(points, observed)
                    self.assertTrue(np.isfinite(result[middle - 1:middle + 2]).all())
                    assert_array_equal(result[observed], points[observed])


if __name__ == "__main__":
    unittest.main()
