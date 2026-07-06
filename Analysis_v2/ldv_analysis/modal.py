"""Modal analysis of LDV cable-coupling datasets.

Treats each loaded dataset (one cable + gap-length configuration) as a
doubly-clamped span and extracts its transverse resonances, complex mode
shapes, theoretical-mode match (MAC), and polarization state.

Numerics only — all plotting lives in :mod:`ldv_analysis.modal_plotting`.

The single entry point is :func:`analyze_dataset`, which stores everything in
``cfg['modal']`` so downstream cells / plots never recompute.

Pipeline (matches the project notebook sections):
    §2  spectral_fingerprint   — |V_y|, |V_z| per-sensor → max / total-power fingerprints
    §2  detect_resonances      — scipy.find_peaks on the combined detection signal
    §3  extract_mode_shape     — narrow bandpass → integrate → complex FFT bin at f_n
    §4  mac / theoretical_modes — clamped-clamped Euler-Bernoulli comparison
    §5  polarization_ellipse   — per-sensor ellipse params + dominant label
"""

import numpy as np
from scipy.signal import find_peaks, welch

from .signal import bandpass, integrate_fft, amplitude_spectrum
from .analysis import prepare_geometry


# ─────────────────────────────────────────────────────────────────────────────
# Theoretical doubly-clamped (clamped-clamped) Euler-Bernoulli beam mode shapes
# ─────────────────────────────────────────────────────────────────────────────
# Roots of cos(α)cosh(α) = 1 — the frequency equation for a beam clamped at both
# ends. The natural frequencies scale as f_n ∝ α_n² · L⁻².
ALPHA_CC = np.array([4.730041, 7.853205, 10.995608, 14.137165, 17.278759])


def clamped_clamped_mode(xi, n):
    """Normalised clamped-clamped beam mode shape n (1-based) at ξ = x/L ∈ [0, 1].

    φ_n(ξ) = [cosh(αξ) − cos(αξ)] − σ_n [sinh(αξ) − sin(αξ)],
    with σ_n = (cosh α − cos α)/(sinh α − sin α) and α = ALPHA_CC[n-1].

    Returned shape is scaled so max|φ| = 1.
    """
    a = ALPHA_CC[n - 1]
    sigma = (np.cosh(a) - np.cos(a)) / (np.sinh(a) - np.sin(a))
    xi = np.asarray(xi, dtype=float)
    phi = (np.cosh(a * xi) - np.cos(a * xi)) - sigma * (np.sinh(a * xi) - np.sin(a * xi))
    peak = np.max(np.abs(phi))
    return phi / peak if peak > 0 else phi


def theoretical_modes(xi, n_modes=5):
    """Stack the first `n_modes` clamped-clamped shapes at positions ξ.

    Returns array of shape (n_modes, len(xi)), each row max-normalised.
    """
    return np.array([clamped_clamped_mode(xi, n) for n in range(1, n_modes + 1)])


def theoretical_fn_ratios(n_modes=5):
    """Frequency ratios f_n / f_1 = (α_n / α_1)² for the clamped-clamped beam."""
    return (ALPHA_CC[:n_modes] / ALPHA_CC[0]) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Modal Assurance Criterion
# ─────────────────────────────────────────────────────────────────────────────
def mac(a, b):
    """Modal Assurance Criterion between two (possibly complex) mode vectors.

    MAC = |aᴴb|² / [(aᴴa)(bᴴb)] ∈ [0, 1].  A global phase / scale on either
    vector cancels, so a complex measured shape can be compared directly with a
    real theoretical shape (node sign-flips appear as ~180° phase and are
    handled correctly).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    num = np.abs(np.vdot(a, b)) ** 2          # vdot conjugates its first argument
    den = float(np.real(np.vdot(a, a) * np.vdot(b, b)))
    return float(num / den) if den > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# §2  Spectral fingerprint + resonance detection
# ─────────────────────────────────────────────────────────────────────────────
def gap_indices(cfg):
    """Sensor indices spanning the hanging (doubly-clamped) gap [idx_left, idx_right]."""
    return np.arange(cfg['idx_left'], cfg['idx_right'] + 1)


def smooth_spectrum(freqs, spec, smooth_hz):
    """Moving-average (boxcar) smoothing of a spectrum over ±smooth_hz/2.

    spec may be (N_f,) or (N_f, N_s); smoothing runs along axis 0. The window
    length is round(smooth_hz / Δf) bins (minimum 1 → no-op). Reduces the
    bin-to-bin fluctuations that make find_peaks fire on noise wiggles.
    """
    spec = np.asarray(spec, dtype=float)
    if smooth_hz is None or smooth_hz <= 0 or len(freqs) < 2:
        return spec
    df = float(freqs[1] - freqs[0])
    n_win = max(1, int(round(smooth_hz / df)))
    if n_win <= 1:
        return spec
    kernel = np.ones(n_win) / n_win
    if spec.ndim == 1:
        return np.convolve(spec, kernel, mode='same')
    out = np.empty_like(spec)
    for j in range(spec.shape[1]):
        out[:, j] = np.convolve(spec[:, j], kernel, mode='same')
    return out


def _welch_amp_spectra(data, fs, nperseg=None):
    """Per-column Welch amplitude spectral density sqrt(PSD) of (N_t, N_s) data.

    nperseg default round(2·fs) → Δf = 0.5 Hz, ~11 averaged segments for the
    12 s sweep records: the averaging itself smooths the spectrum (variance
    ∝ 1/n_segments), which is the natural FFT-parameter route to a stable
    fingerprint.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    if nperseg is None:
        nperseg = int(round(2 * fs))
    nperseg = max(8, min(nperseg, n))
    noverlap = nperseg // 2
    f, P = welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap,
                 window='hann', axis=0)
    return f, np.sqrt(P)


def shaker_input_spectrum(cfg, method='welch', nperseg=None):
    """Amplitude spectra of the three shaker-head velocity components.

    Quantifies the direction the shaker actually excites at each frequency —
    the *_SHAKER.mat recording is taken directly on the shaker head, so any
    energy in v_y / v_z is transverse input the cable sees in addition to the
    intended axial (x) drive.

    Returns dict(freqs, amp_x, amp_y, amp_z,
                 frac_x, frac_y, frac_z)   # power fractions, sum = 1 per bin
    """
    fs = cfg['fs']
    comps = {}
    for key, name in (('sh_vx', 'x'), ('sh_vy', 'y'), ('sh_vz', 'z')):
        sig = np.asarray(cfg[key], dtype=float).reshape(-1, 1)
        if method == 'welch':
            f, a = _welch_amp_spectra(sig, fs, nperseg=nperseg)
        else:
            f, a = amplitude_spectrum(sig, fs)
        comps[name] = a[:, 0]
    p_tot = comps['x'] ** 2 + comps['y'] ** 2 + comps['z'] ** 2 + 1e-300
    return dict(freqs=f,
                amp_x=comps['x'], amp_y=comps['y'], amp_z=comps['z'],
                frac_x=comps['x'] ** 2 / p_tot,
                frac_y=comps['y'] ** 2 / p_tot,
                frac_z=comps['z'] ** 2 / p_tot)


def spectral_fingerprint(cfg, gap_idx=None, method='welch', nperseg=None,
                         smooth_hz=None, normalize_by_input=False):
    """Per-sensor spectra of v_y, v_z → detection fingerprints across sensors.

    method:
        'welch' (default) — per-sensor Welch amplitude spectral density
            (nperseg = round(2·fs) → Δf = 0.5 Hz, ~11 averages). The averaging
            suppresses the bin-to-bin fluctuations that caused spurious peak
            detections with the raw FFT.
        'fft'   — single-record amplitude spectrum (Δf ≈ 0.08 Hz, no averaging;
            the original behaviour).
    smooth_hz : optional extra boxcar smoothing width [Hz] applied to the
        detection signal (both methods).
    normalize_by_input : if True, divide the total-power detection signal by
        the shaker-head input power at each frequency (all three components).
        Peaks that merely mirror a peak in the shaker output — instead of a
        cable resonance — are flattened out; true resonances (output ≫ input)
        remain. Adds keys 'fp_total_raw' and 'shaker_power' to the output.

    Returns a dict with:
        freqs    : (N_f,)  frequency axis [Hz]
        spec_y   : (N_f, N_gap)  |V_y| per gap sensor
        spec_z   : (N_f, N_gap)
        fp_y_max : (N_f,)  max |V_y| across sensors          (per-polarization)
        fp_z_max : (N_f,)  max |V_z| across sensors
        fp_y_pow : (N_f,)  Σ |V_y|² across sensors           (total power)
        fp_z_pow : (N_f,)  Σ |V_z|² across sensors
        fp_total : (N_f,)  detection signal (smoothed / normalized as requested)
    """
    if gap_idx is None:
        gap_idx = gap_indices(cfg)
    fs = cfg['fs']
    if method == 'welch':
        freqs, spec_y = _welch_amp_spectra(cfg['vy'][:, gap_idx], fs, nperseg)
        _, spec_z = _welch_amp_spectra(cfg['vz'][:, gap_idx], fs, nperseg)
    else:
        freqs, spec_y = amplitude_spectrum(cfg['vy'][:, gap_idx], fs)
        _, spec_z = amplitude_spectrum(cfg['vz'][:, gap_idx], fs)

    if smooth_hz:
        spec_y = smooth_spectrum(freqs, spec_y, smooth_hz)
        spec_z = smooth_spectrum(freqs, spec_z, smooth_hz)

    fp_y_max = spec_y.max(axis=1)
    fp_z_max = spec_z.max(axis=1)
    fp_y_pow = (spec_y ** 2).sum(axis=1)
    fp_z_pow = (spec_z ** 2).sum(axis=1)
    fp_total = fp_y_pow + fp_z_pow

    out = dict(
        freqs=freqs, spec_y=spec_y, spec_z=spec_z,
        fp_y_max=fp_y_max, fp_z_max=fp_z_max,
        fp_y_pow=fp_y_pow, fp_z_pow=fp_z_pow,
    )
    if normalize_by_input:
        sh = shaker_input_spectrum(cfg, method=method, nperseg=nperseg)
        p_in = np.interp(freqs, sh['freqs'],
                         sh['amp_x'] ** 2 + sh['amp_y'] ** 2 + sh['amp_z'] ** 2)
        if smooth_hz:
            p_in = smooth_spectrum(freqs, p_in, smooth_hz)
        out['fp_total_raw'] = fp_total
        out['shaker_power'] = p_in
        fp_total = fp_total / (p_in + 1e-300)
    out['fp_total'] = fp_total
    return out


def detect_resonances(freqs, detect_signal, f_min, f_max,
                      prominence, min_spacing_hz):
    """Find resonance frequencies in `detect_signal` via scipy.signal.find_peaks.

    The detection signal is normalised to its in-band maximum, so `prominence`
    is a dimensionless fraction (0–1) of the strongest peak. `min_spacing_hz`
    is converted to the `distance` parameter using the bin width.

    Returns
    -------
    f_peaks   : (N_pk,) peak frequencies [Hz], ascending
    idx_peaks : (N_pk,) indices into the FULL `freqs` array
    sig_norm  : (N_f,)  normalised detection signal (0 outside the band)
    """
    freqs = np.asarray(freqs)
    band = (freqs >= f_min) & (freqs <= f_max)
    s = np.asarray(detect_signal, dtype=float).copy()
    smax = s[band].max() + 1e-30
    sig_norm = np.where(band, s / smax, 0.0)

    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    distance = max(1, int(round(min_spacing_hz / df)))

    idx_peaks, _props = find_peaks(sig_norm, prominence=prominence, distance=distance)
    # find_peaks works on the masked-to-zero array; keep only in-band peaks
    idx_peaks = idx_peaks[band[idx_peaks]]
    order = np.argsort(freqs[idx_peaks])
    idx_peaks = idx_peaks[order]
    return freqs[idx_peaks], idx_peaks, sig_norm


# ─────────────────────────────────────────────────────────────────────────────
# §3  Complex mode-shape extraction at a single resonance
# ─────────────────────────────────────────────────────────────────────────────
def extract_mode_shape(cfg, f_n, gap_idx=None, bw_frac=0.10):
    """Complex modal amplitudes φ_y, φ_z at every gap sensor for resonance f_n.

    1. Narrow bandpass v_y, v_z around f_n·(1 ± bw_frac).
    2. Integrate velocity → displacement (FFT integration).
    3. Take the FFT bin nearest f_n → complex amplitude per sensor.

    Returns
    -------
    phi_y, phi_z : (N_gap,) complex arrays
    f_bin        : float — the actual FFT-bin frequency used [Hz]
    """
    if gap_idx is None:
        gap_idx = gap_indices(cfg)
    fs, dt = cfg['fs'], cfg['dt']
    f_lo = max(f_n * (1.0 - bw_frac), 0.5)
    f_hi = f_n * (1.0 + bw_frac)

    vy_bp = bandpass(cfg['vy'][:, gap_idx], fs, f_lo, f_hi)
    vz_bp = bandpass(cfg['vz'][:, gap_idx], fs, f_lo, f_hi)
    uy = integrate_fft(vy_bp, dt)
    uz = integrate_fft(vz_bp, dt)

    N = uy.shape[0]
    fr = np.fft.rfftfreq(N, d=dt)
    k = int(np.argmin(np.abs(fr - f_n)))
    Uy = np.fft.rfft(uy, axis=0)
    Uz = np.fft.rfft(uz, axis=0)
    return Uy[k, :], Uz[k, :], float(fr[k])


# ─────────────────────────────────────────────────────────────────────────────
# §5  Polarization-ellipse analysis
# ─────────────────────────────────────────────────────────────────────────────
def polarization_ellipse(phi_y, phi_z):
    """Per-sensor polarization-ellipse parameters from complex (φ_y, φ_z).

    Decomposes the planar motion Re([φ_y, φ_z]·e^{iωt}) into co- and
    counter-rotating circular components A₊, A₋:

        A₊ = ½(φ_y + i φ_z),   A₋ = ½(φ_y* + i φ_z*)
        major = |A₊| + |A₋|,   minor = ‖A₊| − |A₋‖
        θ     = ½(arg A₊ + arg A₋)        major-axis orientation in the y–z plane
        ratio = minor / major            (0 = linear, 1 = circular)
        hand  = sign Im(φ_y · φ_z*)      (+1 / −1 rotation sense; |A₊|>|A₋| ⇔ +1)

    All inputs/outputs are arrays of equal shape.
    """
    phi_y = np.asarray(phi_y)
    phi_z = np.asarray(phi_z)
    Ap = 0.5 * (phi_y + 1j * phi_z)
    Am = 0.5 * (np.conj(phi_y) + 1j * np.conj(phi_z))
    major = np.abs(Ap) + np.abs(Am)
    minor = np.abs(np.abs(Ap) - np.abs(Am))
    theta = 0.5 * (np.angle(Ap) + np.angle(Am))
    ratio = minor / (major + 1e-30)
    hand = np.sign(np.imag(phi_y * np.conj(phi_z)))
    return dict(major=major, minor=minor, theta=theta,
                axis_ratio=ratio, handedness=hand)


def classify_polarization(axis_ratio, lin_thresh=0.1, circ_thresh=0.8):
    """Map an axis ratio (or array) to 'linear' / 'elliptical' / 'circular'."""
    a = np.asarray(axis_ratio, dtype=float)
    out = np.where(a < lin_thresh, 'linear',
                   np.where(a > circ_thresh, 'circular', 'elliptical'))
    return out if out.ndim else str(out)


def dominant_polarization(pol, lin_thresh=0.1, circ_thresh=0.8, amp_frac=0.2):
    """Single polarization label for a resonance by majority of significant sensors.

    Only sensors whose major axis exceeds `amp_frac`·max(major) vote (noise-floor
    sensors are excluded). Returns
        (label, dom_axis_ratio, dom_handedness)
    where dom_axis_ratio is the amplitude-weighted mean ratio over voters and
    dom_handedness is the sign of the amplitude-weighted handedness.
    """
    major = pol['major']
    ratio = pol['axis_ratio']
    hand = pol['handedness']
    sig = major > (amp_frac * (major.max() + 1e-30))
    if not np.any(sig):
        sig = np.ones_like(major, dtype=bool)

    labels = classify_polarization(ratio[sig], lin_thresh, circ_thresh)
    vals, counts = np.unique(labels, return_counts=True)
    # tie-break by total major-axis amplitude per class
    weights = {v: major[sig][labels == v].sum() for v in vals}
    best = max(zip(vals, counts), key=lambda vc: (vc[1], weights[vc[0]]))[0]

    w = major[sig]
    dom_ratio = float(np.average(ratio[sig], weights=w))
    dom_hand = int(np.sign(np.average(hand[sig], weights=w)))
    return str(best), dom_ratio, dom_hand


def ellipse_trace(phi_y, phi_z, n=120):
    """Sampled (y, z) trajectory of one oscillation cycle for plotting an ellipse."""
    th = np.linspace(0.0, 2.0 * np.pi, n)
    e = np.exp(1j * th)
    return np.real(phi_y * e), np.real(phi_z * e)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def analyze_dataset(cfg, f_min=5.0, f_max=500.0, prominence=0.05,
                    min_spacing_hz=8.0, bw_frac=0.10, n_theory_modes=5,
                    lin_thresh=0.1, circ_thresh=0.8, verbose=True,
                    fingerprint_method='welch', nperseg=None, smooth_hz=None,
                    normalize_by_input=False):
    """Run the full modal pipeline for one dataset and stash it in cfg['modal'].

    Requires cfg to be loaded (io.load_cable_dataset). Geometry is prepared on
    demand if chord_unit is missing.

    fingerprint_method / nperseg / smooth_hz / normalize_by_input control the
    detection-signal construction (see spectral_fingerprint): Welch averaging
    (default) and optional boxcar smoothing stabilise the peak detection, and
    input normalization divides out the shaker-head excitation spectrum so
    shaker-borne peaks are not misread as cable resonances.

    Returns the cfg['modal'] dict:
        freqs, fp_*            — fingerprints (§2)
        sig_norm, peaks        — detection signal + resonance frequencies (§2)
        gap_idx, xi, s_mm      — gap-sensor positions
        theory_xi, theory      — fine-grid theoretical shapes for overlay (§4)
        modes                  — list of per-resonance dicts:
            f_n, f_bin, phi_y, phi_z,
            mac_vec, mode_order, mac,
            pol (per-sensor ellipse params),
            dom_pol, dom_axis_ratio, dom_handedness
        params                 — the parameters used
    """
    if 'chord_unit' not in cfg:
        prepare_geometry(cfg)

    gap_idx = gap_indices(cfg)
    XYZ = cfg['XYZ']
    e_hat = cfg['chord_unit']
    L = cfg['chord_len']
    s = np.array([np.dot(XYZ[i] - XYZ[cfg['idx_left']], e_hat) for i in gap_idx])
    xi = s / L if L > 0 else s

    fp = spectral_fingerprint(cfg, gap_idx, method=fingerprint_method,
                              nperseg=nperseg, smooth_hz=smooth_hz,
                              normalize_by_input=normalize_by_input)
    f_peaks, idx_peaks, sig_norm = detect_resonances(
        fp['freqs'], fp['fp_total'], f_min, f_max, prominence, min_spacing_hz)

    # Fine grid of theoretical shapes (for smooth overlay curves)
    theory_xi = np.linspace(0.0, 1.0, 200)
    theory = theoretical_modes(theory_xi, n_theory_modes)
    # Theoretical shapes sampled at the actual sensor positions (for MAC)
    theory_at_sensors = theoretical_modes(xi, n_theory_modes)

    modes = []
    for f_n in f_peaks:
        phi_y, phi_z, f_bin = extract_mode_shape(cfg, f_n, gap_idx, bw_frac)
        pol = polarization_ellipse(phi_y, phi_z)
        dom_pol, dom_ratio, dom_hand = dominant_polarization(
            pol, lin_thresh, circ_thresh)

        # MAC against each theoretical mode using the dominant-polarization
        # complex shape (the component carrying the most modal energy).
        phi_dom = phi_y if np.sum(np.abs(phi_y) ** 2) >= np.sum(np.abs(phi_z) ** 2) else phi_z
        mac_vec = np.array([mac(phi_dom, theory_at_sensors[m]) for m in range(n_theory_modes)])
        mode_order = int(np.argmax(mac_vec) + 1)

        modes.append(dict(
            f_n=float(f_n), f_bin=f_bin,
            phi_y=phi_y, phi_z=phi_z,
            dom_component=('y' if phi_dom is phi_y else 'z'),
            mac_vec=mac_vec, mode_order=mode_order, mac=float(mac_vec.max()),
            pol=pol, dom_pol=dom_pol,
            dom_axis_ratio=dom_ratio, dom_handedness=dom_hand,
        ))

    cfg['modal'] = dict(
        **fp,
        sig_norm=sig_norm, peaks=f_peaks, peak_idx=idx_peaks,
        gap_idx=gap_idx, xi=xi, s_mm=s * 1e3,
        theory_xi=theory_xi, theory=theory,
        modes=modes,
        params=dict(f_min=f_min, f_max=f_max, prominence=prominence,
                    min_spacing_hz=min_spacing_hz, bw_frac=bw_frac,
                    n_theory_modes=n_theory_modes,
                    lin_thresh=lin_thresh, circ_thresh=circ_thresh,
                    fingerprint_method=fingerprint_method,
                    smooth_hz=smooth_hz,
                    normalize_by_input=normalize_by_input),
    )
    if verbose:
        print(f"  [{cfg['label']}] {len(modes)} resonance(s): "
              + ", ".join(f"{m['f_n']:.1f}Hz(M{m['mode_order']},{m['dom_pol'][:4]})"
                          for m in modes))
    return cfg['modal']
