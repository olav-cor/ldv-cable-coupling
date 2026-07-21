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

Transfer functions (output relative to input) come in two flavours:
    averaged   H(f) = S_YX(f) / S_XX(f) = csd(X, Y) / welch(X)   ('h1', 'h2', 'hv')
    direct     H(f) = 𝔉{Y}(f) / 𝔉{X}(f)                          ('direct')
With X passed as the first argument to scipy.signal.csd, arg(H) is the phase of
the output relative to the input and |H| is the output/input amplitude ratio
(verified numerically against a known phase lag).
"""

import numpy as np
from scipy.signal import csd, welch
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


def welch_spectra(X, Y, fs, nperseg=None, noverlap=None, window='hann'):
    """Welch auto- and cross-spectra of input X and output Y.

    All three spectra share the same segmentation (nperseg, noverlap, window),
    so every FRF estimator below is built from one consistent set of averages:

        S_XX(f) = ⟨|X_k(f)|²⟩          (welch(X))
        S_YY(f) = ⟨|Y_k(f)|²⟩          (welch(Y))
        S_XY(f) = ⟨X_k*(f) · Y_k(f)⟩   (csd(X, Y); conjugate on the FIRST arg)

    and the ordinary coherence  γ²(f) = |S_XY|² / (S_XX · S_YY).

    Defaults: nperseg = round(2·fs) → Δf = 0.5 Hz at fs = 5 kHz,
    noverlap = nperseg // 2, Hann window.

    Returns (f, Pxx, Pyy, Pxy, coh).
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    nperseg, noverlap = _welch_params(fs, len(X), nperseg, noverlap)
    kw = dict(fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    f, Pxx = welch(X, **kw)
    _, Pyy = welch(Y, **kw)
    _, Pxy = csd(X, Y, **kw)
    coh = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-300)
    return f, Pxx, Pyy, Pxy, coh


def normalise_estimator(estimator):
    """Canonical estimator name: 'h1', 'h2', 'hv' or 'direct'.

    None, 'direct', 'raw' and 'none' all map to 'direct' — the un-averaged
    single-FFT ratio Y(f)/X(f), i.e. no estimator at all (see direct_transfer).
    """
    if estimator is None:
        return 'direct'
    est = str(estimator).lower()
    if est in ('direct', 'raw', 'none'):
        return 'direct'
    if est in ('h1', 'h2', 'hv'):
        return est
    raise ValueError("estimator must be 'h1', 'h2', 'hv' or 'direct'/None, "
                     f"got {estimator!r}")


def estimate_H(Pxx, Pyy, Pxy, estimator='h1'):
    """Averaged FRF estimate from the three Welch spectra.

    estimator:
        'h1' : H₁ = S_XY / S_XX
               Least-squares fit assuming uncorrelated noise on the OUTPUT
               only. Underestimates |H| when the input is noisy.
        'h2' : H₂ = S_YY / S_YX = S_YY / S_XY*
               Assumes noise on the INPUT only. Overestimates |H| when the
               output is noisy; best near resonances where the input is weak.
               |H₁| ≤ |H_true| ≤ |H₂| brackets the FRF, and γ² = |H₁| / |H₂|.
        'hv' : H_v — total least squares, noise on BOTH channels (equal
               noise-power assumed). Smallest-eigenvalue eigenvector of the
               2×2 cross-spectral matrix per frequency bin:

                   λ₋  = (S_XX+S_YY)/2 − √( ((S_XX−S_YY)/2)² + |S_XY|² )
                   H_v = (S_YY − λ₋) / S_XY*

               |H_v| always lies between |H₁| and |H₂|.

    'direct' is not an averaged estimator and cannot be formed from the Welch
    spectra — use direct_transfer(X, Y, fs) on the time series instead (or
    compute_transfer_functions(..., estimator='direct'), which routes it).

    Returns the complex H(f) array.
    """
    est = normalise_estimator(estimator)
    if est == 'h1':
        return Pxy / Pxx
    if est == 'h2':
        return Pyy / np.conj(Pxy)
    if est == 'hv':
        lam_minus = 0.5 * (Pxx + Pyy) - np.sqrt(
            (0.5 * (Pxx - Pyy)) ** 2 + np.abs(Pxy) ** 2)
        return (Pyy - lam_minus) / np.conj(Pxy)
    raise ValueError("estimate_H builds averaged estimators only; for "
                     "estimator='direct' call direct_transfer(X, Y, fs)")


def welch_transfer(X, Y, fs, nperseg=None, noverlap=None, window='hann',
                   estimator='h1'):
    """FRF and coherence between input X and output Y (Welch method).

    H(f) is built from the shared Welch spectra with the chosen estimator
    (see estimate_H: 'h1', 'h2' or 'hv').
    γ²(f) is the ordinary coherence — independent of the estimator choice.

    Defaults: nperseg = round(2·fs), noverlap = nperseg//2, Hann window, H1.

    Returns (f, H (complex), coh).
    """
    f, Pxx, Pyy, Pxy, coh = welch_spectra(X, Y, fs, nperseg=nperseg,
                                          noverlap=noverlap, window=window)
    return f, estimate_H(Pxx, Pyy, Pxy, estimator), coh


def direct_transfer(X, Y, fs, window='hann'):
    """Raw single-record FRF — no averaging, no estimator.

    H_direct(f) = 𝔉{w·Y}(f) / 𝔉{w·X}(f)

    computed from ONE windowed FFT of the full record. Every frequency bin is a
    single realisation: no bias from averaging, but also no variance reduction —
    amplitude and phase scatter wherever excitation or response is weak.
    Included as the baseline the averaged estimators (H1/H2/Hv) improve upon.

    Returns (f, H (complex)) on the full-record FFT grid
    (Δf = fs / N ≈ 0.08 Hz for the 12 s sweep records).
    """
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    n = len(X)
    if window == 'hann':
        w = np.hanning(n)
    elif window in (None, 'boxcar', 'rect'):
        w = np.ones(n)
    else:
        from scipy.signal import get_window
        w = get_window(window, n)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    Xf = np.fft.rfft(X * w)
    Yf = np.fft.rfft(Y * w)
    return f, Yf / np.where(np.abs(Xf) > 0, Xf, np.inf)


def compute_transfer_functions(cfg, signals=None, nperseg=None, noverlap=None,
                               window='hann', estimator='h1',
                               all_estimators=True):
    """Compute H_mid and H_η (+ coherences) for one dataset.

    The primary keys (H_mid, H_eta) use `estimator`, which all downstream η
    summaries then read:

        'h1' | 'h2' | 'hv'   averaged Welch estimators (see estimate_H), on the
                             Welch grid f (Δf = fs / nperseg ≈ 0.5 Hz).
        'direct' or None     the raw single-FFT ratio Y(f)/X(f) — no averaging,
                             no estimator (see direct_transfer) — on the dense
                             full-record FFT grid (Δf = fs / N ≈ 0.08 Hz).

    In direct mode tf['f'] IS the dense grid, so H_eta/H_mid/coh_eta/coh_mid all
    stay mutually consistent and every downstream consumer (band_mean_eta,
    resonance_summary, the FRF figures) works unchanged — only much noisier,
    which is the price of not averaging.

    Coherence in direct mode: γ² is identically 1 for a single realisation and
    therefore carries no information. To keep the usual quality gate meaningful,
    coh_eta/coh_mid are the Welch coherences interpolated onto the dense grid.
    They gate which bins are averaged; they never enter H itself.

    The Welch grid and its coherences are always kept alongside, under
    f_welch / coh_eta_welch / coh_mid_welch, so estimator-comparison code can
    pick the grid a given key lives on (see frf_arrays).

    With all_estimators=True the dict additionally carries every FRF variant:

        H_eta_h1 / H_eta_h2 / H_eta_hv     (Welch grid, f_welch)
        H_mid_h1 / H_mid_h2 / H_mid_hv
        f_direct, H_eta_direct, H_mid_direct   (dense full-record FFT grid)

    Stashes and returns a dict with f, H_mid, coh_mid, H_eta, coh_eta, grid
    ('welch' or 'direct'), the estimator variants and the parameters used.
    Builds signals first if not already present.
    """
    est = normalise_estimator(estimator)
    if signals is None:
        signals = cfg.get('fd_signals') or build_coupling_signals(cfg)
    fs = signals['fs']
    X, Y_eta, Y_mid = signals['X'], signals['Y_eta'], signals['Y_mid']

    # Welch spectra are computed either way: they supply the coherence used for
    # gating (and the h1/h2/hv variants when requested).
    f_w, Pxx, Pyy_e, Pxy_e, coh_eta_w = welch_spectra(
        X, Y_eta, fs, nperseg=nperseg, noverlap=noverlap, window=window)
    _, _, Pyy_m, Pxy_m, coh_mid_w = welch_spectra(
        X, Y_mid, fs, nperseg=nperseg, noverlap=noverlap, window=window)
    nper, nov = _welch_params(fs, len(X), nperseg, noverlap)

    need_direct = (est == 'direct') or all_estimators
    if need_direct:
        f_d, H_eta_d = direct_transfer(X, Y_eta, fs, window=window)
        _, H_mid_d = direct_transfer(X, Y_mid, fs, window=window)

    if est == 'direct':
        f, H_eta, H_mid = f_d, H_eta_d, H_mid_d
        coh_eta = np.interp(f, f_w, coh_eta_w)
        coh_mid = np.interp(f, f_w, coh_mid_w)
        nper_used, nov_used = len(X), 0
    else:
        f = f_w
        H_eta = estimate_H(Pxx, Pyy_e, Pxy_e, est)
        H_mid = estimate_H(Pxx, Pyy_m, Pxy_m, est)
        coh_eta, coh_mid = coh_eta_w, coh_mid_w
        nper_used, nov_used = nper, nov

    tf = dict(f=f, H_mid=H_mid, coh_mid=coh_mid,
              H_eta=H_eta, coh_eta=coh_eta,
              estimator=est, grid='direct' if est == 'direct' else 'welch',
              nperseg=nper_used, noverlap=nov_used, df=float(f[1] - f[0]),
              f_welch=f_w, coh_eta_welch=coh_eta_w, coh_mid_welch=coh_mid_w,
              df_welch=float(f_w[1] - f_w[0]))

    if all_estimators:
        for e in ('h1', 'h2', 'hv'):
            tf[f'H_eta_{e}'] = estimate_H(Pxx, Pyy_e, Pxy_e, e)
            tf[f'H_mid_{e}'] = estimate_H(Pxx, Pyy_m, Pxy_m, e)
    if need_direct:
        tf['f_direct'] = f_d
        tf['H_eta_direct'] = H_eta_d
        tf['H_mid_direct'] = H_mid_d

    cfg['fd_tf'] = tf
    return tf


def frf_arrays(tf, key='H_eta'):
    """(f, H, coh) for any FRF key in tf, on the grid that key actually lives on.

    A tf dict can hold two frequency grids at once (dense direct + Welch), so
    pairing tf['f'] with an arbitrary H key is a shape trap. This resolves it:

        'H_eta' / 'H_mid'          → the primary grid tf['f'] (whichever it is)
        'H_eta_h1' / 'H_mid_hv' …  → the Welch grid tf['f_welch']
        'H_eta_direct' / 'H_mid_direct' → the dense grid tf['f_direct']

    The coherence returned is always the Welch coherence expressed on that grid
    (interpolated for the dense one) — see compute_transfer_functions.
    """
    which = 'eta' if 'eta' in key else 'mid'
    if key in ('H_eta', 'H_mid'):
        return tf['f'], tf[key], tf[f'coh_{which}']
    if key.endswith('_direct'):
        f = tf['f_direct']
        coh = (tf[f'coh_{which}'] if tf.get('grid') == 'direct'
               else np.interp(f, tf['f_welch'], tf[f'coh_{which}_welch']))
        return f, tf[key], coh
    return tf['f_welch'], tf[key], tf[f'coh_{which}_welch']


def band_mean_frf(tf, key='H_eta', f_min=0.0, f_max=50.0, coh_thresh=0.7):
    """Coherence-gated band mean of |H| for any FRF key (grid-aware).

    Returns (mean, std, n_bins); (nan, nan, 0) if no bin passes the gate.
    """
    f, H, coh = frf_arrays(tf, key)
    m = (f >= f_min) & (f <= f_max) & (coh >= coh_thresh)
    if not m.any():
        return np.nan, np.nan, 0
    vals = np.abs(H[m])
    return float(vals.mean()), float(vals.std()), int(m.sum())


def band_mean_eta(cfg, f_min=0.0, f_max=50.0, coh_thresh=0.7, tf=None):
    """Mean strain-transfer efficiency η over a frequency band, FRF method.

    η(f) = |H_η(f)| for whichever estimator the tf was built with (H1/H2/Hv or
    direct). The band average uses only bins with γ²(η) ≥ coh_thresh so
    low-coherence (noise-dominated) bins do not bias the result — in direct mode
    that gate comes from the Welch coherence interpolated onto the dense grid.

    Returns (eta_mean, eta_std, n_bins); (None, None, 0) if no coherent bins.
    Result is cached in cfg['fd_eta_summary'][(f_min, f_max, coh_thresh)].
    """
    if tf is None:
        tf = cfg.get('fd_tf') or compute_transfer_functions(cfg)
    mean, std, n = band_mean_frf(tf, 'H_eta', f_min, f_max, coh_thresh)
    out = (None, None, 0) if n == 0 else (mean, std, n)
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

    Peak picking is a single-bin argmax, so it wants an AVERAGED tf: on a direct
    (un-averaged) tf the global max is easily a lone noise spike and f₁/Q are not
    trustworthy. Pass an H1/Hv tf here even when the η summaries use 'direct'.

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
                         x_source='ends', estimator='h1'):
    """Full frequency-domain pipeline for one dataset (Sections 1, 2, 4, 5).

    Builds signals, computes Welch FRFs + coherence (all estimator variants),
    the STFT point estimates, and the resonance summary. Requires geometry
    prepared (prepare_geometry or plot_initial_geometry). x_source selects the
    δL reference (see build_coupling_signals); estimator picks the primary FRF
    ('h1', 'h2', 'hv', or 'direct'/None for the un-averaged Y(f)/X(f)).
    Returns (signals, tf, stft, resonance).
    """
    signals = build_coupling_signals(cfg, x_source=x_source)
    tf = compute_transfer_functions(cfg, signals, nperseg=nperseg,
                                    noverlap=noverlap, estimator=estimator)
    stft = stft_point_transfers(cfg, stft_freqs, signals=signals)
    res = resonance_summary(cfg, tf, f_max=f_max_res)
    return signals, tf, stft, res
