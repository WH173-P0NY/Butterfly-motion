# Plane construction: source, derivation, and unresolved reconstruction

## Primary source checked

Qiang Chen et al., *A Practical Method for Butterfly Motion Capture*, MIG 2022,
DOI https://doi.org/10.1145/3561975.3562940, section 4.1, PDF pages 3-5.

The ACM landing page returned HTTP 403 during this review. The complete
9-page paper was obtained from the author's site:
https://qiangchen-cg.github.io/assets/paper/A%20Practical%20Method%20for%20Butterfly%20Motion%20Capture.pdf
It carries the same title, authors, ACM publication notice, and DOI. The
equations on PDF page 4 and Figure 5 on PDF page 5 were visually inspected,
not merely read from extracted text. This is an author-hosted copy; byte
identity to the currently ACM-hosted PDF has not been established.

## Two added functions

Both functions return a unit normal. The plane is anchored at the supplied
current thorax position `t` and has equation `dot(n, x - t) = 0`. The caller
passes explicit anatomical positions, including the actual future sample
`t_future = r_t,i+m`; neither function guesses indices or substitutes a
previous sample at the end of a recording. These are per-frame building
blocks, not a complete missing-marker reconstruction pipeline.

### `reference_plane_normal(thorax, future_thorax)`

Source: section 4.1, the paragraph defining the forward direction, Eq. (1),
and Figure 5(a). Eq. (1) constructs the auxiliary point as
`r_o = t + h*j`, with `j = (0,1,0)` and `h = 10`.

With `f = t_future - t`, a normal is parallel to `f cross (h*j)`.
The function returns `normalize(f cross j)`. Omitting positive `h` does not
change the plane or unit normal. This cross product is the geometric
derivation of the plane from its defining points, not an extra fitted model.
The plane's vertical axis is **Y**, exactly as stated alongside Eq. (1).

### `posture_plane_normal(thorax, future_thorax, left_wing, right_wing)`

Source: section 4.1's actual-posture-plane paragraph and Figure 5(b).
The figure shows the plane through `t`, `t_future`, and
`mid = (left_wing + right_wing)/2`. Accordingly the function returns
`normalize(f cross (mid-t))`. The paper does not give a separate numbered
cross-product equation for this construction; this formula is derived from
the plane shown in Figure 5(b).

This is a symmetry/posture plane, **not** the plane through `t`, `left_wing`,
and `right_wing`. A regression test distinguishes those planes. Both wings
must already be observed. This function does not claim to obtain this plane
when one wing is missing; that is the separate roll-interpolation step.

## Explicit numerical/API policies (not paper claims)

- Inputs must be finite real XYZ vectors of shape `(3,)`. Missing positions
  must be resolved or skipped before calling; NaN/Inf raises `ValueError`.
- Zero displacement, zero midpoint offset, nonfinite derived lengths, and
  collinearity raise `ValueError` rather than inventing a plane.
- After normalizing defining directions, a cross-product norm <= `1e-12`
  is considered numerically collinear. This sine threshold is an explicit
  implementation tolerance, not a parameter supplied by the paper.
- Reference construction is undefined for stationary or vertical flight.
  Posture construction is undefined when the midpoint coincides with the
  thorax or lies on the thorax displacement line. A posture plane can still
  exist for vertical flight even when the reference plane is undefined.
- Operand order fixes a reproducible normal sign. The geometric plane is
  unchanged by negating its normal. This convention is **not** a definition
  of the paper's signed roll angle, nor a solution for temporal angle wraps.
- There is no smoothing, plane fitting over multiple frames, anatomical
  calibration, or inference of heading from a stationary marker.

## Verified data format and what is not known

All 18 local TXT files are byte-identical to the author's dataset snapshot
`1cdfe52f25593694b9f73ef84a9d4b063dc7728d` in
https://github.com/ButterflyDataset/butterflymotiondatasets . They have nine
numeric columns, i.e. three contiguous XYZ-sized blocks. However, neither
the inspected README nor the XLS documents which block is thorax/left/right;
the XLS maps flight numbers to butterfly numbers only. Nine columns also do
not provide separate trajectories for all five physical markers described
in section 3. No undocumented column-to-anatomy mapping or merger of the
physical markers is implemented.

The dataset integration test therefore tries all three candidate thorax
blocks and both remaining blocks as wings. It checks geometry invariants
only; it does not validate anatomical identity. It uses two interior frames
per flight and an example step `m=8` (reported in section 4.3, not optimized
by this change): 18 * 3 * 2 = 108 configurations. This is intentionally not
presented as 108 verified butterfly poses or an accuracy experiment.

## Why symmetric reconstruction is not implemented yet

1. **Printed sign inconsistency in Eqs. (2)-(3).** Equation (3) uses the
   signed dot product `(r_r-t) dot n / norm(n)`. Equation (2) adds
   `2*d*n/norm(n)`. Substituting a unit normal and a point with signed
   plane coordinate `s` gives plane coordinate `3*s`, whereas an actual
   reflection must give `-s`. Flipping `n` does not repair this: `d*n` is
   unchanged. For the plane x=0 and r_r=(1,0,0), literal evaluation gives
   (3,0,0), not (-1,0,0). The source does not explain a different sign
   convention resolving this. We do not silently replace '+' with '-'.
2. **Equation (4)'s coordinate frame.** The printed matrix is a row-vector
   rotation in the XY plane (global Z axis if read in world coordinates),
   while Figure 5 describes roll relative to flight direction and Eq. (1)
   defines global +Y as vertical. A local-frame transform and signed-angle
   convention are not specified there. No arbitrary replacement rotation,
   roll unwrapping, or interpolation of ambiguous angles is added.
3. **Anatomical column mapping** remains unverified as described above.

These block a claim of faithful complete reconstruction. They do not block
the two fully observed plane constructions. Clarification or a reference
implementation is needed before claiming to reproduce the reconstruction
and its reported errors.

## Tests and limits of verification

Run from the repository root:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

The added tests check incidence of all defining points, independent analytic
normals, unit length, known tilt, agreement of a zero-roll fixture, translation
and positive-scale invariance, wing swapping, input immutability, invalid
values/shapes, stationary/vertical motion, coincident and nearly collinear
points, and the 108 unlabeled dataset configurations. Existing interpolation
tests are retained. No accuracy against ground-truth normals, recovered
wing markers, complete environment installation, or full pipeline is tested.

The previous `reconstruction.py` placeholder is untouched. New geometry is
kept in a separate valid module; its tests do not import that placeholder.
