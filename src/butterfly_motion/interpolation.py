import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_cubic(
    trajectory: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fill interior gaps in one uniformly sampled XYZ marker trajectory.

    ``trajectory`` must be a nonempty real numeric array of shape (N, 3).
    ``mask`` is an optional boolean array of shape (N,): True means an
    observed point and False means a missing point, even if its stored
    coordinates are finite. With no mask, any row containing NaN or infinity
    is missing. Explicitly observed points must have finite coordinates.

    Return a floating-point copy, preserving observed points exactly.
    Missing points outside the observed frame interval remain NaN: this
    routine does not extrapolate. At least two observed points are required
    if anything is missing. With two points SciPy uses linear interpolation;
    with three, its default not-a-knot spline is a parabola.

    Frame indices are the time coordinate. The routine does not perform
    symmetry-based wing reconstruction, smoothing, or parameter fitting.
    """
    points = np.asarray(trajectory)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("trajectory must have nonempty shape (N, 3)")
    if points.dtype.kind not in "iuf":
        raise ValueError("trajectory must contain real numeric coordinates")

    finite = np.isfinite(points).all(axis=1)
    if mask is None:
        observed = finite
    else:
        observed = np.asarray(mask)
        if observed.shape != (len(points),) or observed.dtype.kind != "b":
            raise ValueError("mask must be a boolean array of shape (N,)")
        if not finite[observed].all():
            raise ValueError("observed points must contain only finite coordinates")

    result = points.astype(float, copy=True)
    missing = ~observed
    if not missing.any():
        return result
    if np.count_nonzero(observed) < 2:
        raise ValueError("Interpolation Error: Not enough points to interpolate")

    frames = np.arange(len(points))
    spline = CubicSpline(
        frames[observed], result[observed], axis=0, extrapolate=False
    )
    result[missing] = spline(frames[missing])
    return result
