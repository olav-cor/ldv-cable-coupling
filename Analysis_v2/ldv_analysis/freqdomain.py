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

The two flavours differ in *two* ways: the estimator algebra, and the amount of
spectral averaging. Welch averages K ≈ 11 time segments; the direct estimate
averages nothing, so it is both denser (Δf = fs/N ≈ 0.08 Hz) and far noisier.
To separate the two effects the direct spectra can be smoothed in the frequency
direction instead (Daniell smoothing, see direct_spectra): a running kernel of
equivalent noise bandwidth B Hz gives the same variance reduction as averaging
B/Δf independent bins, so setting B = Δf_Welch puts the direct estimate at the
same effective resolution as H1/H2/Hv while keeping the dense grid. This is the
default in compute_transfer_functions (direct_smooth_hz='welch').
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


def welch_enbw(fs, nperseg, window='hann'):
    """Equivalent noise bandwidth [Hz] of a Welch estimate — its true resolution.

    A tapered segment does not resolve to its bin spacing Δf = fs/nperseg: the
    taper spreads each line over ENBW = fs·Σw²/(Σw)², which is 1.5·Δf for a Hann
    window (1·Δf only for a boxcar). This is the bandwidth to match when putting
    another estimator "at the same spectral resolution" as Welch.
    """
    w = _taper(int(nperseg), window)
    return float(fs * np.sum(w ** 2) / np.sum(w) ** 2)


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


def _taper(n, window):
    """Time-domain taper of length n ('hann', None/'boxcar'/'rect', or scipy)."""
    if window == 'hann':
        return np.hanning(n)
    if window in (None, 'boxcar', 'rect'):
        return np.ones(n)
    from scipy.signal import get_window
    return get_window(window, n)


def _record_fft(sig, fs, window='hann'):
    """Windowed rFFT of one full record → (f, S(f)). Δf = fs / N."""
    sig = np.asarray(sig, dtype=float).ravel()
    n = len(sig)
    return (np.fft.rfftfreq(n, d=1.0 / fs),
            np.fft.rfft(sig * _taper(n, window)))


def _daniell_kernel(bw_bins, window='hann'):
    """Normalised smoothing kernel with equivalent noise bandwidth ≈ bw_bins.

    A kernel k with Σk = 1 averages n_eff = 1/Σk² *independent* bins; that is
    the quantity setting both the variance reduction and the effective
    resolution bandwidth (boxcar of m bins → n_eff = m, Hann → n_eff ≈ 2m/3).
    Since only odd lengths are usable (an even kernel shifts the spectrum by
    half a bin), the length is chosen by a short search for the odd m whose
    n_eff lands closest to the requested bandwidth.

    Returns (kernel or None, n_eff). None/1.0 means "no smoothing".
    """
    bw = float(bw_bins)
    if not np.isfinite(bw) or bw <= 1.0:
        return None, 1.0

    def _k(m):
        if window in (None, 'boxcar', 'rect'):
            return np.ones(m) / m
        from scipy.signal import get_window
        w = get_window(window, m, fftbins=False)
        return w / w.sum()

    best = None
    for m in range(3, int(3 * bw) + 7, 2):
        k = _k(m)
        n_eff = 1.0 / float(np.sum(k ** 2))
        if best is None or abs(n_eff - bw) < abs(best[1] - bw):
            best = (k, n_eff)
        if n_eff > bw:                       # bandwidth is monotonic in m
            break
    return best


def _smooth_bins(S, kernel):
    """Edge-normalised running average of a spectrum with a Daniell kernel.

    Convolving a ones-array with the same kernel and dividing by it keeps the
    first/last few bins unbiased (plain mode='same' would taper them towards 0).
    Works for real and complex input.
    """
    if kernel is None:
        return S
    norm = np.convolve(np.ones(len(S)), kernel, mode='same')
    if np.iscomplexobj(S):
        out = (np.convolve(S.real, kernel, mode='same')
               + 1j * np.convolve(S.imag, kernel, mode='same'))
    else:
        out = np.convolve(S, kernel, mode='same')
    return out / norm


def direct_spectra(X, Y, fs, window='hann', smooth_hz=0.0, smooth_window='hann'):
    """Single-record auto/cross spectra, optionally smoothed over frequency.

    The single-record analogue of welch_spectra. From ONE windowed FFT of the
    full record (Δf = fs/N ≈ 0.08 Hz) it forms the raw periodogram products

        S_XX = |X(f)|²,   S_YY = |Y(f)|²,   S_XY = X*(f)·Y(f)

    and then averages each of them over neighbouring frequency bins (Daniell
    smoothing) with a kernel of equivalent noise bandwidth `smooth_hz`. Averaging
    over frequency instead of over time segments buys the same variance
    reduction — n_eff ≈ smooth_hz/Δf independent bins per estimate — without
    shortening the record, so the fine 0.08 Hz grid is kept. It is the natural
    choice for a sweep, where every segment of the record carries a different
    frequency band and time-segment averaging is therefore uneven.

    smooth_hz = 0 (default) leaves the raw periodogram: n_eff = 1, coherence is
    identically 1 and carries no information.

    Because the products are smoothed *before* the division, the estimate is
    input-power weighted: bins where |X| is small contribute little instead of
    blowing up, which is what makes the smoothed direct FRF usable.

    Returns (f, Sxx, Syy, Sxy, coh, n_eff). All spectra are unnormalised (the
    window and 1/N factors cancel in every ratio used downstream).
    """
    f, Xf = _record_fft(X, fs, window)
    _, Yf = _record_fft(Y, fs, window)
    return (f,) + _direct_products(f, Xf, Yf, smooth_hz, smooth_window)


def _direct_products(f, Xf, Yf, smooth_hz, smooth_window='hann'):
    """Smoothed periodogram products from two record FFTs → (Sxx, Syy, Sxy,
    coh, n_eff). Shared by direct_spectra and compute_transfer_functions."""
    df = float(f[1] - f[0]) if len(f) > 1 else np.inf
    kernel, n_eff = _daniell_kernel((smooth_hz or 0.0) / df, smooth_window)
    Sxx = _smooth_bins(np.abs(Xf) ** 2, kernel)
    Syy = _smooth_bins(np.abs(Yf) ** 2, kernel)
    Sxy = _smooth_bins(np.conj(Xf) * Yf, kernel)
    coh = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-300)
    return Sxx, Syy, Sxy, coh, n_eff


def direct_transfer(X, Y, fs, window='hann', smooth_hz=0.0, smooth='power',
                    smooth_window='hann'):
    """Single-record FRF — no time averaging, no estimator algebra.

    H_direct(f) = 𝔉{w·Y}(f) / 𝔉{w·X}(f)

    computed from ONE windowed FFT of the full record. With smooth_hz = 0 (the
    default) every frequency bin is a single realisation: no bias from
    averaging, but also no variance reduction — amplitude and phase scatter
    wherever excitation or response is weak. That is the baseline the averaged
    estimators (H1/H2/Hv) improve upon.

    smooth_hz > 0 smooths over frequency instead (see direct_spectra), which
    brings the estimate to the same effective resolution as Welch while keeping
    the dense grid. Two ways to apply it:

        smooth='power' (default) — smooth the periodogram products first,
            H = ⟨X*Y⟩ / ⟨|X|²⟩. Input-power weighted, so weak-input bins cannot
            blow up. Formally this is H1 built by frequency averaging rather
            than segment averaging, and it is the estimate that answers "is the
            H1/direct difference just smoothing?".
        smooth='ratio' — smooth the raw complex ratio itself, H = ⟨Y/X⟩. Keeps
            the literal Y/X definition per bin, but weights every bin equally
            (noisy where |X| is small) and attenuates |H| where the phase turns
            fast, i.e. across resonances.

    Returns (f, H (complex)) on the full-record FFT grid
    (Δf = fs / N ≈ 0.08 Hz for the 12 s sweep records).
    """
    f, Xf = _record_fft(X, fs, window)
    _, Yf = _record_fft(Y, fs, window)
    H_raw = Yf / np.where(np.abs(Xf) > 0, Xf, np.inf)
    if not smooth_hz:
        return f, H_raw
    if smooth == 'ratio':
        df = float(f[1] - f[0]) if len(f) > 1 else np.inf
        kernel, _ = _daniell_kernel(smooth_hz / df, smooth_window)
        return f, _smooth_bins(H_raw, kernel)
    if smooth != 'power':
        raise ValueError(f"smooth must be 'power' or 'ratio', got {smooth!r}")
    Sxx, _, Sxy, _, _ = _direct_products(f, Xf, Yf, smooth_hz, smooth_window)
    return f, Sxy / Sxx


def compute_transfer_functions(cfg, signals=None, nperseg=None, noverlap=None,
                               window='hann', estimator='h1',
                               all_estimators=True, direct_smooth_hz='welch',
                               direct_smooth='power', direct_coh='welch'):
    """Compute H_mid and H_η (+ coherences) for one dataset.

    The primary keys (H_mid, H_eta) use `estimator`, which all downstream η
    summaries then read:

        'h1' | 'h2' | 'hv'   averaged Welch estimators (see estimate_H), on the
                             Welch grid f (Δf = fs / nperseg ≈ 0.5 Hz).
        'direct' or None     the single-FFT ratio Y(f)/X(f) — no time averaging,
                             no estimator algebra (see direct_transfer) — on the
                             dense full-record FFT grid (Δf = fs/N ≈ 0.08 Hz).

    Direct smoothing (direct_smooth_hz) — the resolution knob:
        'welch' (default)  smooth the direct spectra over a band equal to the
                           *resolution bandwidth* of the Welch estimate,
                           ENBW = 1.5·fs/nperseg = 0.75 Hz with the defaults
                           (see welch_enbw — the Hann taper spreads each line
                           over 1.5 bins, so Welch does not actually resolve to
                           its 0.5 Hz bin spacing). That averages
                           n_eff = ENBW/Δf_direct ≈ 9 dense bins per estimate,
                           putting the direct FRF at the same effective
                           resolution and a comparable variance to H1/H2/Hv, so
                           a comparison between them isolates the estimator
                           algebra rather than the amount of averaging.
        'welch_df'         match the nominal Welch bin spacing instead
                           (0.5 Hz → n_eff ≈ 6): slightly sharper, noisier.
        0 / None           no smoothing — the raw single-realisation ratio.
        float              explicit smoothing bandwidth in Hz.
    direct_smooth is 'power' (smooth |X|² and X*Y, then divide — default) or
    'ratio' (smooth Y/X itself); see direct_transfer.

    In direct mode tf['f'] IS the dense grid, so H_eta/H_mid/coh_eta/coh_mid all
    stay mutually consistent and every downstream consumer (band_mean_eta,
    resonance_summary, the FRF figures) works unchanged.

    Coherence in direct mode (direct_coh): γ² is identically 1 for a single
    realisation, so the gate has always come from the Welch coherence
    interpolated onto the dense grid — that stays the default ('welch'). Keeping
    one common quality measure means smoothing the direct estimate changes only
    H, never which bins a band average runs over.
    Smoothing does make a genuine single-record coherence available (n_eff ≈ 9
    bins), stored as coh_eta_dense / coh_mid_dense and selectable with
    direct_coh='dense'. Be aware it is the stricter measure: it sees the noise
    floor of the whole 12 s record in every bin, whereas Welch's segment
    averaging hides it, so on a log sweep it falls off much earlier in frequency
    and gates away substantially more high-frequency bins. Either way the
    coherence only gates which bins are averaged — it never enters H itself.

    The Welch grid and its coherences are always kept alongside, under
    f_welch / coh_eta_welch / coh_mid_welch, so estimator-comparison code can
    pick the grid a given key lives on (see frf_arrays).

    With all_estimators=True the dict additionally carries every FRF variant:

        H_eta_h1 / H_eta_h2 / H_eta_hv     (Welch grid, f_welch)
        H_mid_h1 / H_mid_h2 / H_mid_hv
        f_direct, H_eta_direct, H_mid_direct       (dense grid, smoothed)
        H_eta_direct_raw, H_mid_direct_raw         (dense grid, un-smoothed)
        coh_eta_direct, coh_mid_direct             (dense-grid gate)
        coh_eta_dense, coh_mid_dense               (single-record γ², or None)

    Stashes and returns a dict with f, H_mid, coh_mid, H_eta, coh_eta, grid
    ('welch' or 'direct'), the estimator variants and the parameters used
    (nperseg, noverlap, df, df_welch, df_direct, direct_smooth_hz, direct_n_eff).
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
    df_w = float(f_w[1] - f_w[0])

    # Direct smoothing bandwidth: 'welch' means "match the Welch resolution".
    if isinstance(direct_smooth_hz, str):
        key = direct_smooth_hz.lower()
        if key == 'welch':
            smooth_hz = welch_enbw(fs, nper, window)   # 1.5·Δf for Hann
        elif key == 'welch_df':
            smooth_hz = df_w                            # nominal bin spacing
        else:
            raise ValueError("direct_smooth_hz must be 'welch', 'welch_df', "
                             f"None/0 or a bandwidth in Hz, got "
                             f"{direct_smooth_hz!r}")
    else:
        smooth_hz = float(direct_smooth_hz or 0.0)

    need_direct = (est == 'direct') or all_estimators
    if need_direct:
        f_d, Xf = _record_fft(X, fs, window)
        _, Yf_e = _record_fft(Y_eta, fs, window)
        _, Yf_m = _record_fft(Y_mid, fs, window)
        denom = np.where(np.abs(Xf) > 0, Xf, np.inf)
        H_eta_raw, H_mid_raw = Yf_e / denom, Yf_m / denom

        if smooth_hz > 0:
            Sxx, _, Sxy_e, coh_eta_dense, n_eff = _direct_products(
                f_d, Xf, Yf_e, smooth_hz)
            _, _, Sxy_m, coh_mid_dense, _ = _direct_products(
                f_d, Xf, Yf_m, smooth_hz)
            if direct_smooth == 'power':
                H_eta_d, H_mid_d = Sxy_e / Sxx, Sxy_m / Sxx
            elif direct_smooth == 'ratio':
                kernel, _ = _daniell_kernel(
                    smooth_hz / float(f_d[1] - f_d[0]), 'hann')
                H_eta_d = _smooth_bins(H_eta_raw, kernel)
                H_mid_d = _smooth_bins(H_mid_raw, kernel)
            else:
                raise ValueError("direct_smooth must be 'power' or 'ratio', "
                                 f"got {direct_smooth!r}")
        else:
            H_eta_d, H_mid_d, n_eff = H_eta_raw, H_mid_raw, 1.0
            coh_eta_dense = coh_mid_dense = None   # γ² ≡ 1, no information

        # Gate on the dense grid. Default stays the interpolated Welch
        # coherence: it is the same quality measure the Welch estimators are
        # gated with, so switching the direct estimate on/off (or smoothing it)
        # never moves the set of bins a band average runs over. The
        # single-record γ² is stored alongside for inspection.
        coh_eta_i = np.interp(f_d, f_w, coh_eta_w)
        coh_mid_i = np.interp(f_d, f_w, coh_mid_w)
        if direct_coh == 'dense' and coh_eta_dense is not None:
            coh_eta_d, coh_mid_d = coh_eta_dense, coh_mid_dense
        elif direct_coh in ('welch', 'dense'):
            coh_eta_d, coh_mid_d = coh_eta_i, coh_mid_i
        else:
            raise ValueError("direct_coh must be 'welch' or 'dense', got "
                             f"{direct_coh!r}")

    if est == 'direct':
        f, H_eta, H_mid = f_d, H_eta_d, H_mid_d
        coh_eta, coh_mid = coh_eta_d, coh_mid_d
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
              df_welch=df_w)

    if all_estimators:
        for e in ('h1', 'h2', 'hv'):
            tf[f'H_eta_{e}'] = estimate_H(Pxx, Pyy_e, Pxy_e, e)
            tf[f'H_mid_{e}'] = estimate_H(Pxx, Pyy_m, Pxy_m, e)
    if need_direct:
        tf.update(f_direct=f_d,
                  H_eta_direct=H_eta_d, H_mid_direct=H_mid_d,
                  H_eta_direct_raw=H_eta_raw, H_mid_direct_raw=H_mid_raw,
                  coh_eta_direct=coh_eta_d, coh_mid_direct=coh_mid_d,
                  coh_eta_dense=coh_eta_dense, coh_mid_dense=coh_mid_dense,
                  df_direct=float(f_d[1] - f_d[0]),
                  direct_smooth_hz=float(smooth_hz),
                  direct_smooth=direct_smooth if smooth_hz > 0 else None,
                  direct_coh=direct_coh, direct_n_eff=float(n_eff))

    cfg['fd_tf'] = tf
    return tf


def frf_arrays(tf, key='H_eta'):
    """(f, H, coh) for any FRF key in tf, on the grid that key actually lives on.

    A tf dict can hold two frequency grids at once (dense direct + Welch), so
    pairing tf['f'] with an arbitrary H key is a shape trap. This resolves it:

        'H_eta' / 'H_mid'          → the primary grid tf['f'] (whichever it is)
        'H_eta_h1' / 'H_mid_hv' …  → the Welch grid tf['f_welch']
        'H_eta_direct' / 'H_mid_direct' / '…_direct_raw'
                                   → the dense grid tf['f_direct']

    The coherence returned is the one belonging to that grid: the Welch
    coherence on the Welch grid, and on the dense grid either the smoothed
    dense-grid γ² or the interpolated Welch one, whichever
    compute_transfer_functions stored.
    """
    which = 'eta' if 'eta' in key else 'mid'
    if key in ('H_eta', 'H_mid'):
        return tf['f'], tf[key], tf[f'coh_{which}']
    if 'direct' in key:
        f = tf['f_direct']
        coh = tf.get(f'coh_{which}_direct')
        if coh is None:                       # tf from an older run
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

    Peak picking is a single-bin argmax, so it wants a tf on the Welch grid.
    Smoothing has taken the lone-noise-spike failure out of the direct estimate,
    but its dense grid still resolves the sub-1 Hz region below the sweep start,
    where |H_mid| can run away and steal the argmax. Pass an H1/Hv tf here even
    when the η summaries use 'direct'.

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
                         x_source='ends', estimator='h1',
                         direct_smooth_hz='welch'):
    """Full frequency-domain pipeline for one dataset (Sections 1, 2, 4, 5).

    Builds signals, computes Welch FRFs + coherence (all estimator variants),
    the STFT point estimates, and the resonance summary. Requires geometry
    prepared (prepare_geometry or plot_initial_geometry). x_source selects the
    δL reference (see build_coupling_signals); estimator picks the primary FRF
    ('h1', 'h2', 'hv', or 'direct'/None for the single-FFT Y(f)/X(f)).
    direct_smooth_hz sets the frequency smoothing of the direct estimate —
    'welch' (default) matches the Welch resolution, 0 leaves it raw.
    Returns (signals, tf, stft, resonance).
    """
    signals = build_coupling_signals(cfg, x_source=x_source)
    tf = compute_transfer_functions(cfg, signals, nperseg=nperseg,
                                    noverlap=noverlap, estimator=estimator,
                                    direct_smooth_hz=direct_smooth_hz)
    stft = stft_point_transfers(cfg, stft_freqs, signals=signals)
    res = resonance_summary(cfg, tf, f_max=f_max_res)
    return signals, tf, stft, res
