"""Strain estimation: spatial-gradient (finite differences) and 3-D arc-length per segment."""

import numpy as np

from .geometry import project_onto_chord


def strain_spatial_gradient(XYZ, ux, uy, uz, chord_unit=None, direction="chord"):
    """Strain via non-uniform central differences along the cable.

    direction = "cartesian" : ∂u_x / ∂x          (mixes transverse motion if cable not horizontal)
    direction = "chord"     : ∂(u·ê) / ∂s        (correct projection — recommended)

    Returns (N_t, N_s) with NaN at the two array edges.
    """
    if direction == "chord":
        assert chord_unit is not None, "chord_unit required for direction='chord'"
        u_axial = project_onto_chord(ux, uy, uz, chord_unit)
        s = np.array([np.dot(XYZ[i] - XYZ[0], chord_unit) for i in range(len(XYZ))])
    else:
        u_axial = ux
        s = XYZ[:, 0]

    strain = np.full_like(u_axial, np.nan)
    for i in range(1, len(s) - 1):
        ds_p = s[i + 1] - s[i]
        ds_m = s[i] - s[i - 1]
        strain[:, i] = (
            u_axial[:, i + 1] * ds_m / (ds_p * (ds_p + ds_m))
            + u_axial[:, i] * (ds_p - ds_m) / (ds_p * ds_m)
            - u_axial[:, i - 1] * ds_p / (ds_m * (ds_p + ds_m))
        )
    return strain


def strain_3d_arclength(XYZ, ux, uy, uz, idx_left, idx_right):
    """Per-segment 3-D arc-length elongation (Method 2 — linearised).

    For each segment i → i+1 inside the gap,
        δd_i(t) ≈ ê_i · (u_{i+1}(t) - u_i(t))
    and local strain estimate δd_i / L0_i.

    Returns
    -------
    delta_d    : (N_t, N_seg)  per-segment elongation [m]
    L0_seg     : (N_seg,)      reference segment lengths [m]
    seg_strain : (N_t, N_seg)  per-segment strain (δd_i / L0_i)
    delta_xl   : (N_t,)        total cable elongation (sum of δd_i)
    """
    n_seg = idx_right - idx_left
    n_t = ux.shape[0]
    delta_d = np.zeros((n_t, n_seg))
    L0_seg = np.zeros(n_seg)
    seg_strain = np.zeros((n_t, n_seg))

    for k, i in enumerate(range(idx_left, idx_right)):
        seg_vec = XYZ[i + 1] - XYZ[i]
        L0 = np.linalg.norm(seg_vec)
        e_hat_i = seg_vec / L0
        du = np.stack([ux[:, i + 1] - ux[:, i],
                       uy[:, i + 1] - uy[:, i],
                       uz[:, i + 1] - uz[:, i]], axis=1)
        delta_d[:, k] = du @ e_hat_i
        L0_seg[k] = L0
        seg_strain[:, k] = delta_d[:, k] / L0

    delta_xl = delta_d.sum(axis=1)
    return delta_d, L0_seg, seg_strain, delta_xl
