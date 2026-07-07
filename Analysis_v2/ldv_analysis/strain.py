"""Strain estimation: finite-difference gradient, 3-D arc-length, and Fourier-domain gradient."""

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


def strain_fourier_gradient(XYZ, ux, uy, uz, chord_unit, n_uniform=None):
    """Strain via Fourier-domain spatial differentiation on a uniform grid.

    Projects displacement onto the chord axis, resamples onto a uniform
    spatial grid of n_uniform points, computes ∂u/∂s = IFFT(ik · FFT(u)),
    then interpolates back to the original sensor positions.

    The Fourier derivative is spectrally accurate for periodic band-limited
    signals. For finite cables the implicit periodicity assumption can produce
    Gibbs-like edge artefacts; use only gap-interior stations for the
    elongation integral.

    Parameters
    ----------
    XYZ        : (N_s, 3)   sensor positions [m]
    ux, uy, uz : (N_t, N_s) displacement components [m]
    chord_unit : (3,)        unit vector along the chord
    n_uniform  : int, optional
        Number of uniform grid points (default = N_s).

    Returns
    -------
    strain : (N_t, N_s)  axial strain at original sensor positions [m/m]
    """
    from scipy.interpolate import interp1d

    u_axial = project_onto_chord(ux, uy, uz, chord_unit)
    s = np.array([np.dot(XYZ[i] - XYZ[0], chord_unit) for i in range(len(XYZ))])

    # `interp1d(kind='cubic')` requires a strictly increasing grid with at least
    # four unique sample locations. Some geometries project nearly duplicate
    # positions onto the chord, so sort and collapse repeated coordinates first.
    order = np.argsort(s)
    s = s[order]
    u_axial = u_axial[:, order]

    s_unique, inverse = np.unique(s, return_inverse=True)
    if s_unique.size != s.size:
        u_unique = np.zeros((u_axial.shape[0], s_unique.size), dtype=u_axial.dtype)
        counts = np.bincount(inverse)
        for j in range(s_unique.size):
            u_unique[:, j] = u_axial[:, inverse == j].mean(axis=1)
        s = s_unique
        u_axial = u_unique

    N_s = len(s)
    if n_uniform is None:
        n_uniform = N_s

    s_uni = np.linspace(s[0], s[-1], n_uniform)
    ds = s_uni[1] - s_uni[0]
    k = np.fft.fftfreq(n_uniform, d=ds) * 2.0 * np.pi  # [rad/m]

    # Interpolate all time steps simultaneously; u_axial.T has shape (N_s, N_t)
    kind = 'cubic' if s.size >= 4 else 'linear'
    interp_fwd = interp1d(s, u_axial.T, axis=0, kind=kind,
                          fill_value='extrapolate', assume_sorted=True)
    u_uni = interp_fwd(s_uni)  # (n_uniform, N_t)

    # Fourier differentiation: ∂u/∂s  ↔  ik · FFT(u)
    U_k = np.fft.fft(u_uni, axis=0)
    strain_uni = np.fft.ifft(1j * k[:, np.newaxis] * U_k, axis=0).real  # (n_uniform, N_t)

    # Map back to original sensor positions
    interp_back = interp1d(s_uni, strain_uni, axis=0, kind=kind,
                           fill_value='extrapolate', assume_sorted=True)
    return interp_back(s).T  # (N_t, N_s)


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
