"""Cable geometry: chord direction, static sag, projection onto chord."""

import numpy as np


def compute_chord(XYZ, idx_left, idx_right):
    """Return (chord unit vector, chord length, start point)."""
    P1, P2 = XYZ[idx_left], XYZ[idx_right]
    chord = P2 - P1
    L = np.linalg.norm(chord)
    return chord / L, L, P1


def measure_sag(XYZ, idx_left, idx_right, use_3d=True, use_parabola=False):
    """Maximum perpendicular distance of any cable point from the chord.

    Parameters
    ----------
    XYZ         : (N_s, 3) array of sensor positions [m].
    idx_left    : int — left endpoint sensor index.
    idx_right   : int — right endpoint sensor index.
    use_3d      : if True (default), compute perpendicular distances in full
                  3-D x-y-z space.  If False, project cable onto the x-z plane
                  and compute 2-D perpendicular distances there.
    use_parabola: if True, fit a parabola through the cable points first and
                  return the peak of the fitted curve as the sag.  This
                  mitigates the influence of outlier sensor positions.
                  The index returned always refers to the original sensor
                  nearest to the parabola vertex.

    Returns
    -------
    sag_max  : float — peak perpendicular distance [m]
    idx_max  : int   — sensor index where the max occurs (nearest to parabola
                       vertex when use_parabola=True)
    perp     : (N_s,) — per-sensor perpendicular distance (raw, not fitted;
                         0 outside [idx_left, idx_right])
    """
    e_hat, _L, P1 = compute_chord(XYZ, idx_left, idx_right)
    pts = XYZ[idx_left:idx_right + 1]
    n = len(pts)

    if use_3d:
        # Full 3-D perpendicular distances from the chord line
        s_vals = np.array([np.dot(pts[i] - P1, e_hat) for i in range(n)])
        perp_vals = np.array([
            np.linalg.norm((pts[i] - P1) - s_vals[i] * e_hat)
            for i in range(n)
        ])
    else:
        # x-z plane only — chord and distances in 2-D
        dx = XYZ[idx_right, 0] - XYZ[idx_left, 0]
        dz = XYZ[idx_right, 2] - XYZ[idx_left, 2]
        L_xz = np.hypot(dx, dz)
        if L_xz == 0:
            L_xz = 1.0
        ex, ez = dx / L_xz, dz / L_xz
        P1_xz = np.array([XYZ[idx_left, 0], XYZ[idx_left, 2]])
        s_vals = np.zeros(n)
        perp_vals = np.zeros(n)
        for i, p in enumerate(pts):
            v = np.array([p[0], p[2]]) - P1_xz
            s_vals[i] = v[0] * ex + v[1] * ez
            # 2-D unsigned perpendicular distance (cross product magnitude)
            perp_vals[i] = abs(v[0] * ez - v[1] * ex)

    # Build full output array (zeros outside the gap)
    perp_full = np.zeros(len(XYZ))
    perp_full[idx_left:idx_right + 1] = perp_vals

    if use_parabola and n >= 3:
        # Constrained parabola through both endpoints: perp(s) = a * s * (s - L)
        # Endpoints lie on the chord by definition, so perp(0) = perp(L) = 0.
        # This is a 1-parameter linear regression on phi = s*(s-L).
        L_s = s_vals[-1]
        phi = s_vals * (s_vals - L_s)
        denom = float(np.dot(phi, phi))
        a = float(np.dot(phi, perp_vals) / denom) if denom > 0 else 0.0
        if a < 0:
            # a < 0 → perp positive in interior → physical sag
            s_vertex = L_s / 2.0
            sag_max = float(-a * L_s ** 2 / 4.0)
        else:
            # No downward arch — fall back to raw max
            sag_max = float(perp_vals.max())
            s_vertex = s_vals[int(np.argmax(perp_vals))]
        local_idx_max = int(np.argmin(np.abs(s_vals - s_vertex)))
    else:
        local_idx_max = int(np.argmax(perp_vals))
        sag_max = float(perp_vals[local_idx_max])

    idx_max = idx_left + local_idx_max
    return sag_max, idx_max, perp_full


def sag_fit_diagnostics(XYZ, idx_left, idx_right):
    """Parabola-fit diagnostics for the sag measurement (3-D perpendicular).
    ...
    """
    e_hat, _L, P1 = compute_chord(XYZ, idx_left, idx_right)
    pts = XYZ[idx_left:idx_right + 1]
    n = len(pts)
    s_vals = np.array([np.dot(pts[i] - P1, e_hat) for i in range(n)])

    # raw perpendicular VECTORS (not yet reduced to a magnitude)
    perp_vecs = np.array([
        (pts[i] - P1) - s_vals[i] * e_hat for i in range(n)
    ])
    perp_vals = np.linalg.norm(perp_vecs, axis=1)

    L_s = s_vals[-1]
    phi = s_vals * (s_vals - L_s)
    denom = float(np.dot(phi, phi))
    a = float(np.dot(phi, perp_vals) / denom) if denom > 0 else 0.0
    fit = a * phi
    residuals = perp_vals - fit
    sag_fit = float(-a * L_s ** 2 / 4.0) if a < 0 else float(perp_vals.max())

    # --- NEW: recover the sag-plane direction n_hat, then reconstruct  ---
    # --- the fitted parabola as real 3-D points for plotting.         ---
    # n_hat: dominant direction of the perpendicular vectors (robust to
    # noise/twist via SVD, rather than a single point's direction).
    U, S, Vt = np.linalg.svd(perp_vecs, full_matrices=False)
    n_hat = Vt[0]
    # sign convention: make it point the same way as the mean deflection
    if np.dot(perp_vecs.mean(axis=0), n_hat) < 0:
        n_hat = -n_hat

    fit_points = P1 + np.outer(s_vals, e_hat) + np.outer(fit, n_hat)      # (n,3)
    w0_point   = P1 + (L_s / 2.0) * e_hat + sag_fit * n_hat               # (3,)

    return dict(
        sag=sag_fit,
        sag_raw=float(perp_vals.max()),
        a=a,
        residuals=residuals,
        rms_residual=float(np.sqrt(np.mean(residuals ** 2))),
        s=s_vals, perp=perp_vals,
        n_hat=n_hat,
        fit_points=fit_points,     # <- plot these (e.g. [:,0] vs [:,2]) instead of `fit`
        w0_point=w0_point,         # <- plot this for the sag/star marker
    )


def sag_method_label(use_3d, use_parabola):
    """Return a short human-readable label for the active sag method."""
    base = '3D' if use_3d else 'x-z plane'
    suffix = ' + parabola fit' if use_parabola else ''
    return base + suffix


def project_onto_chord(ux, uy, uz, chord_unit):
    """Project 3-D displacement onto the chord direction.

    ux, uy, uz : (N_t,) or (N_t, N_s) arrays
    chord_unit : (3,) unit vector

    Returns the projected scalar displacement u_axial.
    """
    return ux * chord_unit[0] + uy * chord_unit[1] + uz * chord_unit[2]
