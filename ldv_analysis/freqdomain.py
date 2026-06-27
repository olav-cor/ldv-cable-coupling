"""Frequency-domain coupling analysis: Welch transfer functions, coherence,
STFT-per-frequency cross-check, and resonance characterisation.

This module builds the three broadband displacement signals used to reproduce
Fig. 3 of Cable_coupling_v1_Simone.pdf and quantifies the ground-to-cable
strain-transfer FRF.

Signals (per dataset, full sweep record, displacement units [m]):
    X(t)    = δL(t)        chord-projected shaker displacement (the input)
    Y_eta(t)= δxₗ(t)       per-segment 3-D arc-length elongation (axial output)
    Y_mid(t)= u_⊥,mid(t)   midpoint transverse displacement, signed along the
                           static-sag direction (the bending-mode output)

Transfer functions (H1 estimator, output relative to input):
    H(f) = S_YX(f) / S_XX(f) = csd(X, Y) / welch(X)
With X passed as the first argument to scipy.signal.csd, arg(H) is the phase of
the output relative to the input and |H| is the output/input amplitude ratio
(verified numerically against a known phase lag).
"""

import numpy as np
from scipy.signal import csd, welch, coherence
from scipy.optimize import curve_fit

from .config import (N_CYCLES_PER_WINDOW, sweep_time_of_frequency,
                     cable_section_AI, _E_stats, CABLE_PROPERTIES)
from .signal import integrate_fft
from .geometry import project_onto_chord
from .strain import strain_3d_arclength


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — signal construction
# ─────────────────────────────────────────────────────────────────────────────
def midpoint_index(cfg):
    """Sensor index closest to the chord midpoint (s = L_chord / 2).

    Uses the static chord geometry; more robust than the raw sag-apex index
    for nearly-taut cables. Requires cfg['chord_unit'] (run prepare_geometry).
    """
    XYZ, e = cfg['XYZ'], cfg['chord_unit']
    il, ir = cfg['idx_left'], cfg['idx_right']
    s = np.array([np.dot(XYZ[i] - XYZ[il], e) for i in range(len(XYZ))])
    s_mid = 0.5 * (s[il] + s[ir])
    local = il + int(np.argmin(np.abs(s[il:ir + 1] - s_mid)))
    return local


def static_sag_direction(cfg, idx_mid):
    """Unit vector along the static sag at the midpoint, perpendicular to chord.

    Defined as the perpendicular offset of the midpoint sensor from the chord
    line. For a nearly-taut cable (negligible static offset) this is degenerate,
    so we fall back to the gravity direction (−z) projected perpendicular to the
    chord, which is the direction the cable would sag under its own weight.

    Returns (e_sag (3,), used_fallback bool).
    """
    XYZ, e = cfg['XYZ'], cfg['chord_unit']
    il = cfg['idx_left']
    d = XYZ[idx_mid] - XYZ[il]
    perp = d - np.dot(d, e) * e
    n = np.linalg.norm(perp)
    # Threshold: 1 µm of static perpendicular offset.
    if n > 1e-6:
        return perp / n, False
    g = np.array([0.0, 0.0, -1.0])               # gravity points along −z
    g_perp = g - np.dot(g, e) * e
    ng = np.linalg.norm(g_perp)
    if ng < 1e-12:                                # chord aligned with gravity
        g = np.array([0.0, -1.0, 0.0])
        g_perp = g - np.dot(g, e) * e
        ng = np.linalg.norm(g_perp)
    return g_perp / ng, True


def build_coupling_signals(cfg, idx_mid=None, x_source='ends'):
    """Build the three broadband displacement signals X, Y_eta, Y_mid.

    Integrates the full-record shaker and cable velocities to displacement (no
    bandpass — we want the broadband response for Welch averaging), then forms:

        X     = δL(t)         span-change input (see x_source below)
        Y_eta = δxₗ(t)        Σ per-segment 3-D arc-length elongation
        Y_mid = u_⊥,mid(t)    midpoint displacement along the static-sag direction

    x_source selects the reference δL(t) — the input the FRF is measured against:
        'ends'   (default) — cable-native span change between the two endpoint
                  sensors: u_axial(idx_right) − u_axial(idx_left), both projected
                  on the chord. This is δL measured directly on the cable and
                  shares its endpoints with δxₗ, so η isolates bending stress
                  relief. No sign flip needed (chord points left→right, so a
                  positive value is an elongation).
        'shaker' — shaker displacement projected on the chord (the imposed ground
                  motion); negated for shaker_end == 'right' so positive = stretch.
        'left_sensor' — absolute chord-axial displacement of idx_left ONLY. This
                  is NOT a span change and only approximates δL if the opposite
                  end is effectively fixed; provided for inspection.

    Requires cfg loaded (velocities present) and geometry prepared
    (chord_unit, idx_left/right). Stashes the result in cfg['fd_signals'].
    """
    fs, dt, e = cfg['fs'], cfg['dt'], cfg['chord_unit']
    il, ir = cfg['idx_left'], cfg['idx_right']

    # Cable displacement field u(x, t).
    ux = integrate_fft(cfg['vx'], dt)
    uy = integrate_fft(cfg['vy'], dt)
    uz = integrate_fft(cfg['vz'], dt)

    # X = δL(t): the reference span-change input.
    if x_source == 'shaker':
        u_sh_x = integrate_fft(cfg['sh_vx'], dt)
        u_sh_y = integrate_fft(cfg['sh_vy'], dt)
        u_sh_z = integrate_fft(cfg['sh_vz'], dt)
        X = project_onto_chord(u_sh_x, u_sh_y, u_sh_z, e)
        if cfg['shaker_end'] == 'right':
            X = -X
    elif x_source == 'ends':
        u_axial_ends = project_onto_chord(
            ux[:, [il, ir]], uy[:, [il, ir]], uz[:, [il, ir]], e)
        X = u_axial_ends[:, 1] - u_axial_ends[:, 0]   # right − left = elongation
    elif x_source == 'left_sensor':
        X = project_onto_chord(ux[:, il], uy[:, il], uz[:, il], e)
    else:
        raise ValueError(
            f"x_source must be 'ends', 'shaker' or 'left_sensor', got {x_source!r}")

    # Y_eta = δxₗ(t): per-segment 3-D arc-length elongation summed over the gap.
    _dd, _L0, _seg, delta_xl = strain_3d_arclength(
        cfg['XYZ'], ux, uy, uz, il, ir)

    # Y_mid = midpoint transverse displacement along the static-sag direction.
    if idx_mid is None:
        idx_mid = midpoint_index(cfg)
    e_sag, used_fallback = static_sag_direction(cfg, idx_mid)
    # u_mid · e_sag equals the perpendicular component along e_sag, since
    # e_sag ⊥ e_chord by construction. Sign is meaningful (+ = further sag).
    Y_mid = (ux[:, idx_mid] * e_sag[0]
             + uy[:, idx_mid] * e_sag[1]
             + uz[:, idx_mid] * e_sag[2])

    signals = dict(
        fs=fs, t=cfg['t'],
        X=X, Y_eta=delta_xl, Y_mid=Y_mid, x_source=x_source,
        idx_mid=idx_mid, e_sag=e_sag, sag_dir_fallback=used_fallback,
    )
    cfg['fd_signals'] = signals
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Welch transfer functions and coherence
# ─────────────────────────────────────────────────────────────────────────────
def _welch_params(fs, n, nperseg, noverlap):
    if nperseg is None:
        nperseg = int(round(2 * fs))
    nperseg = max(8, min(nperseg, n))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = max(0, min(noverlap, nperseg - 1))
    return nperseg, noverlap


def welch_transfer(X, Y, fs, nperseg=None, noverlap=None, window='hann'):
    """H1 transfer function and coherence between input X and output Y.

    H(f) = S_YX / S_XX = csd(X, Y) / welch(X)   (output relative to input)
    γ²(f) = coherence(X, Y)

    Defaults: nperseg = round(2·fs), noverlap = nperseg//2, Hann window.

    Returns (f, H (complex), coh).
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    nperseg, noverlap = _welch_params(fs, len(X), nperseg, noverlap)
    kw = dict(fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    f, Pxx = welch(X, **kw)
    _, Pxy = csd(X, Y, **kw)
    _, coh = coherence(X, Y, **kw)
    H = Pxy / Pxx
    return f, H, coh


def compute_transfer_functions(cfg, signals=None, nperseg=None, noverlap=None,
                               window='hann'):
    """Compute H_mid and H_η (+ coherences) for one dataset.

    Stashes and returns a dict with f, H_mid, coh_mid, H_eta, coh_eta and the
    Welch parameters used. Builds signals first if not already present.
    """
    if signals is None:
        signals = cfg.get('fd_signals') or build_coupling_signals(cfg)
    fs = signals['fs']
    f, H_mid, coh_mid = welch_transfer(signals['X'], signals['Y_mid'], fs,
                                       nperseg=nperseg, noverlap=noverlap,
                                       window=window)
    _, H_eta, coh_eta = welch_transfer(signals['X'], signals['Y_eta'], fs,
                                       nperseg=nperseg, noverlap=noverlap,
                                       window=window)
    nper, nov = _welch_params(fs, len(signals['X']), nperseg, noverlap)
    tf = dict(f=f, H_mid=H_mid, coh_mid=coh_mid,
              H_eta=H_eta, coh_eta=coh_eta,
              nperseg=nper, noverlap=nov, df=float(f[1] - f[0]))
    cfg['fd_tf'] = tf
    return tf


def band_mean_eta(cfg, f_min=0.0, f_max=50.0, coh_thresh=0.7, tf=None):
    """Mean strain-transfer efficiency η over a frequency band, FRF method.

    η(f) = |H_η(f)| (the Welch strain-transfer amplitude). The band average uses
    only bins with γ²(η) ≥ coh_thresh so low-coherence (noise-dominated) bins do
    not bias the result.

    Returns (eta_mean, eta_std, n_bins); (None, None, 0) if no coherent bins.
    Result is cached in cfg['fd_eta_summary'][(f_min, f_max, coh_thresh)].
    """
    if tf is None:
        tf = cfg.get('fd_tf') or compute_transfer_functions(cfg)
    f = tf['f']
    m = (f >= f_min) & (f <= f_max) & (tf['coh_eta'] >= coh_thresh)
    if not m.any():
        out = (None, None, 0)
    else:
        vals = np.abs(tf['H_eta'][m])
        out = (float(vals.mean()), float(vals.std()), int(m.sum()))
    summary = cfg.setdefault('fd_eta_summary', {})
    summary[(f_min, f_max, coh_thresh)] = dict(
        eta_mean=out[0], eta_std=out[1], n_bins=out[2])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — STFT-per-frequency cross-check
# ─────────────────────────────────────────────────────────────────────────────
def stft_point_transfer(cfg, f_target, signals=None,
                        n_cycles=N_CYCLES_PER_WINDOW):
    """Single-frequency FRF estimate from a windowed FFT of the sweep.

    Extracts a window of ±n_cycles/f_target centred on the sweep arrival time
    t_c(f_target), applies a Hann taper, FFTs X, Y_mid and Y_eta, and reads the
    bin closest to f_target to form the complex point transfer functions.

    Returns dict(f_target, f_bin, H_mid, H_eta) or None if the window is too
    short.
    """
    if signals is None:
        signals = cfg.get('fd_signals') or build_coupling_signals(cfg)
    t, fs = signals['t'], signals['fs']

    t_c = sweep_time_of_frequency(f_target)
    half = n_cycles / f_target
    mask = (t >= t_c - half) & (t <= t_c + half)
    if int(mask.sum()) < 16:
        return None

    win = np.hanning(int(mask.sum()))
    freqs = np.fft.rfftfreq(int(mask.sum()), d=1.0 / fs)
    ib = int(np.argmin(np.abs(freqs - f_target)))

    def _bin(sig):
        return np.fft.rfft(sig[mask] * win)[ib]

    Xb = _bin(signals['X'])
    H_mid = _bin(signals['Y_mid']) / Xb
    H_eta = _bin(signals['Y_eta']) / Xb
    return dict(f_target=float(f_target), f_bin=float(freqs[ib]),
                H_mid=H_mid, H_eta=H_eta)


def stft_point_transfers(cfg, freqs, signals=None,
                         n_cycles=N_CYCLES_PER_WINDOW):
    """Run stft_point_transfer over a list of target frequencies.

    Returns a dict with arrays f, H_mid, H_eta (skipping any window too short).
    """
    if signals is None:
        signals = cfg.get('fd_signals') or build_coupling_signals(cfg)
    out_f, out_mid, out_eta = [], [], []
    for fq in freqs:
        r = stft_point_transfer(cfg, fq, signals=signals, n_cycles=n_cycles)
        if r is None:
            continue
        out_f.append(r['f_bin'])
        out_mid.append(r['H_mid'])
        out_eta.append(r['H_eta'])
    res = dict(f=np.array(out_f),
               H_mid=np.array(out_mid), H_eta=np.array(out_eta))
    cfg['fd_stft'] = res
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — resonance characterisation
# ─────────────────────────────────────────────────────────────────────────────
def _sdof_magnitude(f, gain, f0, Q):
    """Driven damped single-DOF displacement-response magnitude.

    |H(f)| = gain · f0² / sqrt((f0² − f²)² + (f0 f / Q)²)
    Peaks near f0 with |H(f0)| ≈ gain · Q (for moderate-to-high Q).
    """
    return gain * f0 ** 2 / np.sqrt((f0 ** 2 - f ** 2) ** 2 + (f0 * f / Q) ** 2)


def fit_lorentzian(f, mag, f0_guess, half_width=None):
    """Fit an SDOF resonance to |H_mid|(f) near f0_guess → quality factor Q.

    Fits within [f0_guess − W, f0_guess + W] (W = half_width, default
    max(3 Hz, 0.3·f0_guess)). Returns dict(f0, Q, zeta, gain, fit_f, fit_mag,
    success). ζ = 1/(2Q).
    """
    f = np.asarray(f, dtype=float)
    mag = np.asarray(mag, dtype=float)
    if half_width is None:
        half_width = max(3.0, 0.3 * f0_guess)
    band = (f >= f0_guess - half_width) & (f <= f0_guess + half_width)
    fb, mb = f[band], mag[band]
    if len(fb) < 4:
        return dict(f0=f0_guess, Q=np.nan, zeta=np.nan, gain=np.nan,
                    fit_f=fb, fit_mag=mb, success=False)

    peak = mb.max()
    p0 = [peak / max(f0_guess, 1.0), f0_guess, 10.0]
    bounds = ([0.0, fb.min(), 0.5], [np.inf, fb.max(), 1e4])
    try:
        popt, _ = curve_fit(_sdof_magnitude, fb, mb, p0=p0, bounds=bounds,
                            maxfev=20000)
        gain, f0, Q = popt
        fit_f = np.linspace(fb.min(), fb.max(), 400)
        return dict(f0=float(f0), Q=float(Q), zeta=float(1.0 / (2.0 * Q)),
                    gain=float(gain), fit_f=fit_f,
                    fit_mag=_sdof_magnitude(fit_f, *popt), success=True)
    except (RuntimeError, ValueError):
        return dict(f0=f0_guess, Q=np.nan, zeta=np.nan, gain=np.nan,
                    fit_f=fb, fit_mag=mb, success=False)


def theoretical_f1(cfg, coeff=22.4):
    """Theoretical fundamental flexural frequency (Probst/Simone Eq. 18).

    ω₁ = (coeff / L²)·√(EI / ρA),  f₁ = ω₁ / 2π
    coeff = 22.4 = (4.730)² for a clamped–clamped beam (the model's assumption).
    L = chord_len. Uses CABLE_PROPERTIES (mean E). Returns f₁ [Hz] or None if
    ρ/E are not set.
    """
    cable = cfg.get('cable')
    props = CABLE_PROPERTIES.get(cable)
    if props is None or props.get('rho') is None or props.get('E') is None:
        return None
    A, I = cable_section_AI(cable)
    E_mean, _ = _E_stats(cable)
    rho = props['rho']
    L = cfg['chord_len']
    omega1 = coeff / L ** 2 * np.sqrt(E_mean * I / (rho * A))
    return float(omega1 / (2.0 * np.pi))


def resonance_summary(cfg, tf=None, f_max=300.0, half_width=None):
    """Characterise the midpoint resonance f₁ for one dataset.

    f₁ is the global max of |H_mid(f)| below f_max. A Lorentzian fit around it
    yields Q and ζ. The theoretical clamped-beam f₁ is added for comparison.

    Returns dict(label, f1_meas, Q, zeta, f1_theory, theta_pred, eta_pred, fit).
    """
    if tf is None:
        tf = cfg.get('fd_tf') or compute_transfer_functions(cfg)
    f, H_mid = tf['f'], tf['H_mid']
    band = (f > 0) & (f <= f_max)
    fb, mb = f[band], np.abs(H_mid[band])
    i_peak = int(np.argmax(mb))
    f1_meas = float(fb[i_peak])
    fit = fit_lorentzian(f, np.abs(H_mid), f1_meas, half_width=half_width)
    out = dict(
        label=cfg.get('label'),
        f1_meas=f1_meas,
        f1_fit=fit['f0'] if fit['success'] else np.nan,
        Q=fit['Q'], zeta=fit['zeta'],
        f1_theory=theoretical_f1(cfg),
        theta_pred=cfg.get('theta_pred'),
        eta_pred=cfg.get('eta_pred'),
        fit=fit,
    )
    cfg['fd_resonance'] = out
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience driver
# ─────────────────────────────────────────────────────────────────────────────
def run_frequency_domain(cfg, stft_freqs=(5, 10, 20, 50, 100, 200, 300),
                         f_max_res=300.0, nperseg=None, noverlap=None,
                         x_source='ends'):
    """Full frequency-domain pipeline for one dataset (Sections 1, 2, 4, 5).

    Builds signals, computes Welch FRFs + coherence, the STFT point estimates,
    and the resonance summary. Requires geometry prepared (prepare_geometry or
    plot_initial_geometry). x_source selects the δL reference (see
    build_coupling_signals). Returns (signals, tf, stft, resonance).
    """
    signals = build_coupling_signals(cfg, x_source=x_source)
    tf = compute_transfer_functions(cfg, signals, nperseg=nperseg,
                                    noverlap=noverlap)
    stft = stft_point_transfers(cfg, stft_freqs, signals=signals)
    res = resonance_summary(cfg, tf, f_max=f_max_res)
    return signals, tf, stft, res
