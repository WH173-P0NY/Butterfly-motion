# Cubic marker interpolation

This change repairs the existing `interpolate_cubic` function on top of
`marker_interpolation`. It is the cubic interpolation component of section 4.1
of *A Practical Method for Butterfly Motion Capture*, DOI
10.1145/3561975.3562940. It does not implement the separate symmetry-based
single-wing reconstruction or the full processing pipeline.

## Contract and choices

- Input: one nonempty `(N, 3)` real XYZ trajectory, uniformly sampled.
- Existing calls with a boolean mask still work: `True` means observed,
  `False` means missing, including intentionally hidden finite samples.
- The mask is now optional. Without it, a nonfinite coordinate marks the
  entire XYZ point as missing. With an explicit mask, observed points must
  be finite; inconsistent input raises `ValueError`.
- The function returns an independent floating-point copy and does not
  modify its arguments. Observed coordinates are preserved.
- Missing points inside the observed interval are evaluated using SciPy's
  default not-a-knot cubic spline. With two observations this becomes linear;
  with three it becomes quadratic.
- Boundary gaps remain three NaNs, even for finite masked placeholders.
  No extrapolation is attempted. This is an explicit project policy for
  cases without surrounding observations, not a claim about paper behavior.
- Missing data with fewer than two observations raises `ValueError`.
  Complete input, including a single frame, needs no interpolation and is
  copied unchanged.
- Frame indices supply the time coordinate. Irregular timestamps, gap-length
  limits, outlier rejection, and physical motion constraints are not handled.

The repository TXT data has nine columns. Apply the function independently
to each contiguous XYZ block (`[:, 0:3]`, `[:, 3:6]`, `[:, 6:9]`). This component
does not assign anatomical labels or infer missing-data masks from zeros.
It is not a recommendation to use plain cubic interpolation for all wing
occlusions: the paper uses a separate symmetry method for one missing wing.

## Verification

From the repository root, without installing/building the project:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

Only NumPy and SciPy are needed; the tests use Python's standard-library
`unittest`. Tests cover cubic recovery, consecutive gaps, finite masked
samples, NaN/Inf detection, boundary gaps, two/three observations, complete
and insufficient data, invalid inputs, and input immutability. A dataset
smoke test reads all 18 files and 54 XYZ blocks and preserves observations.
The existing missing points in this dataset snapshot occur at a boundary,
so an additional test deliberately hides three interior points in each XYZ
block and checks that the function fills them with finite coordinates while
preserving observations. These are integration checks, not an accuracy
benchmark; exact recovery is tested separately on known cubic trajectories.

The existing unrelated `reconstruction.py` TODO is intentionally untouched.
Full package installation, GUI/CLI integration, and agreement with the paper's
reported errors are not validated by these tests.
