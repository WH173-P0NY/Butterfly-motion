"""Geometry invariants, paper-construction checks, and explicit singularities."""

from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from butterfly_motion.geometry import posture_plane_normal, reference_plane_normal


class PlaneTests(unittest.TestCase):
    def setUp(self):
        self.t = np.array([1., 2., 3.])
        self.future = self.t + [4., 0., 0.]
        self.left = self.t + [0., 2., 3.]
        self.right = self.t + [0., 2., -3.]

    def test_reference_eq1_defining_points_lie_on_plane(self):
        t = np.array([10., 20., 30.])
        future = t + [2., 3., 4.]
        normal = reference_plane_normal(t, future)
        assert_allclose(normal, np.array([-4., 0., 2.]) / np.sqrt(20))
        self.assertAlmostEqual(np.linalg.norm(normal), 1.)
        self.assertAlmostEqual(np.dot(normal, future - t), 0.)
        for h in (10., 1., 100.):
            self.assertAlmostEqual(np.dot(normal, np.array([0., h, 0.])), 0.)

    def test_posture_figure5b_contains_forward_and_midpoint(self):
        t = self.t
        forward = np.array([2., 3., 4.])
        mid = t + [0., 5., 1.]
        left, right = mid + [2., -1., 3.], mid - [2., -1., 3.]
        normal = posture_plane_normal(t, t + forward, left, right)
        # Independent solution: (2,3,4) cross (0,5,1) = (-17,-2,10).
        assert_allclose(normal, np.array([-17., -2., 10.]) / np.sqrt(393))
        self.assertAlmostEqual(np.dot(normal, forward), 0.)
        self.assertAlmostEqual(np.dot(normal, (left + right)/2 - t), 0.)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.)

    def test_posture_is_not_plane_through_current_three_markers(self):
        normal = posture_plane_normal(self.t, self.future, self.left, self.right)
        assert_allclose(normal, [0., 0., 1.])
        self.assertNotEqual(np.dot(normal, self.left - self.t), 0.)
        self.assertNotEqual(np.dot(normal, self.right - self.t), 0.)

    def test_reference_and_posture_agree_for_zero_roll_fixture(self):
        assert_allclose(reference_plane_normal(self.t, self.future),
                        posture_plane_normal(self.t, self.future, self.left, self.right))

    def test_known_tilt_about_flight_direction(self):
        theta = 0.4
        mid = self.t + [0., 2*np.cos(theta), 2*np.sin(theta)]
        # Span is perpendicular to the intended symmetry plane.
        span = np.array([0., -np.sin(theta), np.cos(theta)])
        normal = posture_plane_normal(self.t, self.future, mid + span, mid - span)
        assert_allclose(normal, span, atol=1e-14)

    def test_translation_and_positive_scale_invariance(self):
        inputs = (self.t, self.future, self.left, self.right)
        for scale in (0.001, 1000.):
            for shift in (np.zeros(3), np.array([20., -40., 15.])):
                with self.subTest(scale=scale, shift=shift):
                    args = [p*scale + shift for p in inputs]
                    assert_allclose(reference_plane_normal(*args[:2]), [0, 0, 1])
                    assert_allclose(posture_plane_normal(*args), [0, 0, 1])

    def test_swapping_wings_preserves_posture_plane(self):
        assert_allclose(posture_plane_normal(self.t, self.future, self.left, self.right),
                        posture_plane_normal(self.t, self.future, self.right, self.left))

    def test_reference_rejects_stationary_vertical_and_near_vertical_motion(self):
        for direction in ([0., 0., 0.], [0., 1., 0.], [0., -1., 0.], [1e-14, 1., 0.]):
            with self.subTest(direction=direction):
                with self.assertRaises(ValueError):
                    reference_plane_normal(np.zeros(3), np.array(direction))

    def test_posture_rejects_coincident_and_collinear_defining_points(self):
        for future, left, right in (
            (self.t, self.left, self.right),
            (self.future, self.t + [0, 0, 1], self.t - [0, 0, 1]),
            (self.future, self.t + [2, 0, 1], self.t + [2, 0, -1]),
            (self.future, self.t + [2, 1e-14, 1], self.t + [2, 1e-14, -1]),
        ):
            with self.subTest(future=future, left=left, right=right):
                with self.assertRaises(ValueError):
                    posture_plane_normal(self.t, future, left, right)

    def test_posture_can_exist_when_vertical_reference_is_undefined(self):
        t = np.zeros(3)
        future = np.array([0., 1., 0.])
        with self.assertRaises(ValueError):
            reference_plane_normal(t, future)
        assert_allclose(posture_plane_normal(t, future, [1, 0, 1], [1, 0, -1]), [0, 0, -1])

    def test_invalid_inputs_are_rejected_for_each_argument(self):
        inputs = [self.t, self.future, self.left, self.right]
        for function, count in ((reference_plane_normal, 2), (posture_plane_normal, 4)):
            for index in range(count):
                for invalid in ([1, 2], [1, np.nan, 3], [1, np.inf, 3],
                                [1+2j, 2, 3], ["x", "y", "z"], [[1, 2, 3]]):
                    with self.subTest(function=function.__name__, index=index, invalid=invalid):
                        args = inputs[:count].copy()
                        args[index] = invalid
                        with self.assertRaises(ValueError):
                            function(*args)

    def test_inputs_are_not_modified_and_result_is_independent(self):
        args = [self.t, self.future, self.left, self.right]
        before = [p.copy() for p in args]
        for result in (reference_plane_normal(*args[:2]), posture_plane_normal(*args)):
            self.assertEqual(result.shape, (3,))
            self.assertTrue(all(not np.shares_memory(result, p) for p in args))
        for actual, expected in zip(args, before):
            assert_array_equal(actual, expected)

    def test_dataset_shape_and_plane_invariants_without_claiming_marker_labels(self):
        # The author does not document anatomical column labels. Exercise every
        # possible thorax block and both wing blocks, WITHOUT selecting a mapping.
        root = Path(__file__).resolve().parents[1] / "data" / "raw"
        checked = 0
        for path in sorted(root.glob("*.txt")):
            data = np.loadtxt(path).reshape(-1, 3, 3)
            for thorax_block in range(3):
                wings = [block for block in range(3) if block != thorax_block]
                # m=8 is an example from section 4.3, not a fitted parameter here.
                for frame in (0, len(data)//2):
                    t, future = data[frame, thorax_block], data[frame+8, thorax_block]
                    left, right = data[frame, wings[0]], data[frame, wings[1]]
                    with self.subTest(file=path.name, candidate=thorax_block, frame=frame):
                        ref = reference_plane_normal(t, future)
                        actual = posture_plane_normal(t, future, left, right)
                        assert_allclose(np.linalg.norm(ref), 1.)
                        assert_allclose(np.linalg.norm(actual), 1.)
                        assert_allclose(np.dot(ref, future-t), 0., atol=1e-10)
                        assert_allclose(np.dot(actual, future-t), 0., atol=1e-10)
                        assert_allclose(np.dot(actual, (left+right)/2-t), 0., atol=1e-10)
                        checked += 1
        self.assertEqual(checked, 108)


if __name__ == "__main__":
    unittest.main()
