"""Plane normals from section 4.1 / Figure 5 of DOI 10.1145/3561975.3562940.

Inputs are explicitly named XYZ positions, not unlabeled dataset columns.
Both planes pass through ``thorax``; their equation is
``dot(normal, position - thorax) == 0``. Normal orientation is a local
cross-product convention, not the paper's unspecified signed-roll convention.
"""

import numpy as np


def reference_plane_normal(
    thorax: np.ndarray,
    future_thorax: np.ndarray,
) -> np.ndarray:
    """Return the unit normal of the zero-roll vertical reference plane.

    Section 4.1, Eq. (1), Fig. 5(a): the plane contains r_t,i,
    r_t,i+m, and r_o = r_t,i + 10 * (0, 1, 0). Its normal is therefore
    parallel to (r_t,i+m - r_t,i) cross (0, 1, 0); the positive factor 10
    cancels on normalization. The caller must supply the actual i+m sample.

    Positions must be finite real arrays of shape (3,). Raise ValueError
    for zero displacement or a direction parallel/nearly parallel to +Y
    (cross-product sine <= 1e-12). Do not invent a fallback plane.
    """
    raw = np.asarray([thorax, future_thorax])
    if raw.shape != (2, 3) or raw.dtype.kind not in "iuf":
        raise ValueError("thorax positions must be real XYZ vectors of shape (3,)")
    points = raw.astype(float)
    if not np.isfinite(points).all():
        raise ValueError("thorax positions must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        forward = points[1] - points[0]
        length = np.linalg.norm(forward)
    if not np.isfinite(length) or length == 0:
        raise ValueError("thorax displacement must have a finite nonzero length")
    normal = np.cross(forward / length, [0.0, 1.0, 0.0])
    sine = np.linalg.norm(normal)
    if sine <= 1e-12:
        raise ValueError("vertical displacement does not define a reference plane")
    return normal / sine


def posture_plane_normal(
    thorax: np.ndarray,
    future_thorax: np.ndarray,
    left_wing: np.ndarray,
    right_wing: np.ndarray,
) -> np.ndarray:
    """Return the unit normal of the fully observed posture plane.

    Section 4.1, Fig. 5(b): construct the plane through r_t,i,
    r_t,i+m and r_mid=(r_l,i+r_r,i)/2. The cross-product formula is a
    geometric derivation from that figure, not a numbered paper equation.
    It is NOT the plane through the three current markers.

    All inputs must be finite real XYZ vectors. Missing wings are rejected:
    Eq. (4)'s interpolated-roll reconstruction is not implemented here.
    Coincident/collinear defining points (cross-product sine <= 1e-12)
    raise ValueError. Normal sign is not a signed roll-angle definition.
    """
    raw = np.asarray([thorax, future_thorax, left_wing, right_wing])
    if raw.shape != (4, 3) or raw.dtype.kind not in "iuf":
        raise ValueError("marker positions must be real XYZ vectors of shape (3,)")
    points = raw.astype(float)
    if not np.isfinite(points).all():
        raise ValueError("marker positions must be finite; both wings are required")
    with np.errstate(over="ignore", invalid="ignore"):
        forward = points[1] - points[0]
        midpoint = points[2] / 2 + points[3] / 2
        mid_direction = midpoint - points[0]
        lengths = np.array([np.linalg.norm(forward), np.linalg.norm(mid_direction)])
    if not np.isfinite(lengths).all() or (lengths == 0).any():
        raise ValueError("plane directions must have finite nonzero lengths")
    normal = np.cross(forward / lengths[0], mid_direction / lengths[1])
    sine = np.linalg.norm(normal)
    if sine <= 1e-12:
        raise ValueError("collinear points do not define a posture plane")
    return normal / sine
