import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_cubic(
        trajectory: np.ndarray,
        mask: np.ndarray
) -> np.ndarray:
    """
    Interpolates missing points 
    args
    trajectory: numpy array of 3d points (N,3) where N ~ number of points
    mask: numpy array of bools tell which points to interpolate false point to interpolate, true real point
    """
    frames = np.arange(len(trajectory))
    known_frames = frames[mask]
    known_points = trajectory[mask]

    if len(known_frames) < 2:
        raise ValueError("Interpolation Error: Not enough points to interpolate")
    
    spline = CubicSpline(
        known_frames,
        known_points,
        axis = 0,
        extrapolate = False,
    )

    result = trajectory.copy().astype(float)
    result[~mask] = spline[frames[~mask]]
    
    return result
