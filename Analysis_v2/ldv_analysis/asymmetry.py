"""Extension / compression asymmetry of the cable response.

Numerical simulations show that a sagged segment responds differently when its
endpoints are pulled apart (extension — the sag is straightened, second-order
stiffening) than when they are pushed together (compression — the sag deepens,
the segment softens). For a perfectly linear system the response to the two
half-cycles of a sinusoidal drive is exactly mirror-symmetric; any measurable
difference is a signature of this geometric nonlinearity.

Three complementary detectors are implemented, all operating on the same
windowed → bandpassed → integrated displacement signals as
``analysis.per_frequency_qc`` (so results are directly comparable with the QC
and η notebooks):

1. **Half-cycle conditional η** (``eta_ext`` / ``eta_comp``)
   Split the samples of the reference elongation δ_ref(t) by sign:
   extension (δ_ref > 0, endpoints moving apart) vs compression (δ_ref < 0).
   Median of η(t) = δxₗ(t)/δ_ref(t) over each subset. A linear system gives
   η_ext = η_comp.

2. **Peak-amplitude asymmetry** (``peak_asym``)
   From the cable elongation δxₗ(t): mean positive-peak amplitude A₊ vs mean
   |negative-peak| amplitude A₋ →  α = (A₊ − A₋)/(A₊ + A₋) ∈ [−1, 1].
   α > 0 means the cable elongates further during extension half-cycles than
   it shortens during compression (and vice versa).

3. **Even-harmonic distortion** (``h2_ratio``)
   A response y(t) = c₁·x(t) + c₂·x(t)² + … to a tone x = A·cos(ωt) puts the
   quadratic (sign-asymmetric) term at 2ω (and DC). The ratio
   |Y(2f)| / |Y(f)| of the windowed FFT of δxₗ therefore quantifies the
   asymmetric part of the response independently of the reference signal.
   (Odd/cubic nonlinearity — symmetric softening/stiffening — appears at 3f
   instead and is reported as ``h3_ratio`` for comparison.)

All three are returned per frequency by :func:`asymmetry_at_frequency` and
swept over the band by :func:`asymmetry_sweep`.
"""

import numpy as np
from scipy.signal import hilbert, find_peaks

from .config import (TARGET_FREQS, N_CYCLES_PER_WINDOW,
                     BANDWIDTH_FRAC, BANDWIDTH_ABS_HZ,
                     sweep_time_of_frequency)
from .signal import bandpass, integrate_fft
from .geometry import project_onto_chord
from .strain import strain_3d_arclength
from .analysis import _bp_edges


# Fields returned per frequency (used to build aligned sweep arrays).
ASYM_KEYS = ('f', 'eta_ext', 'eta_comp', 'eta_ratio',
             'peak_pos', 'peak_neg', 'peak_asym',
             'h2_ratio', 'h3_ratio', 'n_ext', 'n_comp')


def _windowed_signals(cfg, f_target, n_cycles=N_CYCLES_PER_WINDOW,
                      bw_frac=BANDWIDTH_FRAC, bw_abs_hz=BANDWIDTH_ABS_HZ):
    """Window → bandpass → integrate, returning (t_win, delta_xl, delta_ends).

    Same pipeline as analysis.per_frequency_qc steps 1–4 (Method 2 elongation
    and endpoint-pair reference). Returns None if the window is too short.
    """
    fs, dt, t = cfg['fs'], cfg['dt'], cfg['t']
    t_c = sweep_time_of_frequency(f_target)
    half = n_cycles / f_target
    mask = (t >= max(t[0], t_c - half)) & (t <= min(t[-1], t_c + half))
    if int(mask.sum()) < 64:
        return None

    f_lo, f_hi = _bp_edges(f_target, bw_frac, bw_abs_hz)
    ux = integrate_fft(bandpass(cfg['vx'], fs, f_lo, f_hi)[mask], dt)
    uy = integrate_fft(bandpass(cfg['vy'], fs, f_lo, f_hi)[mask], dt)
    uz = integrate_fft(bandpass(cfg['vz'], fs, f_lo, f_hi)[mask], dt)

    il, ir = cfg['idx_left'], cfg['idx_right']
    _dd, _L0, _seg, delta_xl = strain_3d_arclength(cfg['XYZ'], ux, uy, uz, il, ir)

    u_ax = project_onto_chord(ux[:, [il, ir]], uy[:, [il, ir]],
                              uz[:, [il, ir]], cfg['chord_unit'])
    delta_ends = u_ax[:, 1] - u_ax[:, 0]        # >0 = endpoints moving apart
    return t[mask], delta_xl, delta_ends


def _harmonic_ratios(sig, fs, f_target):
    """|Y(2f)|/|Y(f)| and |Y(3f)|/|Y(f)| from a Hann-windowed FFT of sig.

    Harmonic amplitudes are read as the RMS FFT magnitude within ±10 % of the
    harmonic frequency (the log-sweep smears each tone over a few bins).
    """
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    win = np.hanning(n)
    Y = np.abs(np.fft.rfft(sig * win))
    fr = np.fft.rfftfreq(n, d=1.0 / fs)

    def _amp(fh):
        b = (fr >= 0.9 * fh) & (fr <= 1.1 * fh)
        return float(np.sqrt(np.mean(Y[b] ** 2))) if b.any() else np.nan

    a1 = _amp(f_target)
    if not np.isfinite(a1) or a1 <= 0:
        return np.nan, np.nan
    return _amp(2 * f_target) / a1, _amp(3 * f_target) / a1


def asymmetry_at_frequency(cfg, f_target, n_cycles=N_CYCLES_PER_WINDOW,
                           bw_frac=BANDWIDTH_FRAC, bw_abs_hz=BANDWIDTH_ABS_HZ,
                           floor_frac=0.25):
    """Extension/compression asymmetry metrics at one swept frequency.

    floor_frac gates the half-cycle split: only samples with
    |δ_ref| > floor_frac · max|δ_ref| vote, so the zero crossings (where η is
    ill-conditioned) are excluded.

    Returns a dict with the keys in ASYM_KEYS (scalars), or None if the window
    is too short. Requires geometry prepared (cfg['chord_unit']).
    """
    out = _windowed_signals(cfg, f_target, n_cycles=n_cycles,
                            bw_frac=bw_frac, bw_abs_hz=bw_abs_hz)
    if out is None:
        return None
    _t, delta_xl, delta_ref = out

    # 1) Half-cycle conditional η -------------------------------------------
    floor = floor_frac * np.abs(delta_ref).max()
    ext = delta_ref > floor          # endpoints moving apart (extension)
    comp = delta_ref < -floor        # endpoints moving together (compression)
    eta_t = delta_xl / np.where(np.abs(delta_ref) > floor, delta_ref, np.nan)
    eta_ext = float(np.nanmedian(eta_t[ext])) if ext.any() else np.nan
    eta_comp = float(np.nanmedian(eta_t[comp])) if comp.any() else np.nan
    eta_ratio = eta_ext / eta_comp if (np.isfinite(eta_comp)
                                       and eta_comp != 0) else np.nan

    # 2) Peak-amplitude asymmetry of the cable elongation --------------------
    # Minimum peak spacing: half a period of the target tone.
    dist = max(1, int(0.5 * cfg['fs'] / f_target))
    prom = 0.2 * np.abs(delta_xl).max()
    pk_pos, _ = find_peaks(delta_xl, distance=dist, prominence=prom)
    pk_neg, _ = find_peaks(-delta_xl, distance=dist, prominence=prom)
    peak_pos = float(np.mean(delta_xl[pk_pos])) if len(pk_pos) else np.nan
    peak_neg = float(np.mean(-delta_xl[pk_neg])) if len(pk_neg) else np.nan
    if np.isfinite(peak_pos) and np.isfinite(peak_neg) and (peak_pos + peak_neg) > 0:
        peak_asym = (peak_pos - peak_neg) / (peak_pos + peak_neg)
    else:
        peak_asym = np.nan

    # 3) Harmonic distortion --------------------------------------------------
    h2, h3 = _harmonic_ratios(delta_xl, cfg['fs'], f_target)

    return dict(f=float(f_target),
                eta_ext=eta_ext, eta_comp=eta_comp, eta_ratio=eta_ratio,
                peak_pos=peak_pos, peak_neg=peak_neg, peak_asym=peak_asym,
                h2_ratio=h2, h3_ratio=h3,
                n_ext=int(ext.sum()), n_comp=int(comp.sum()))


def asymmetry_sweep(cfg, freqs=None, verbose=False, **kw):
    """Run asymmetry_at_frequency over a frequency grid.

    Returns a dict of aligned 1-D arrays (keys in ASYM_KEYS); frequencies with
    too-short windows are skipped. Cached in cfg['asym_sweep'].
    """
    if freqs is None:
        freqs = TARGET_FREQS
    cached = cfg.get('asym_sweep')
    if cached is not None and len(cached['f']) == len(list(freqs)):
        return cached

    acc = {k: [] for k in ASYM_KEYS}
    for f in freqs:
        d = asymmetry_at_frequency(cfg, f, **kw)
        if d is None:
            continue
        for k in ASYM_KEYS:
            acc[k].append(d[k])
        if verbose:
            print(f"  {f:5.1f} Hz  η_ext={d['eta_ext']:.3f}  "
                  f"η_comp={d['eta_comp']:.3f}  peak_asym={d['peak_asym']:+.3f}  "
                  f"H2/H1={d['h2_ratio']:.3f}")
    out = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
    cfg['asym_sweep'] = out
    return out


def linearity_segment_asymmetry(cfg, f_target, res=None, n_amp_bins=12,
                                floor_frac=0.25):
    """Extension/compression η vs drive amplitude within one linearity segment.

    Uses the cached linearity result (cfg['lin_results'][f_target], from
    analysis.run_linearity_analysis) whose amplitude ramps 0→1 over the
    segment: η_ext and η_comp are evaluated in n_amp_bins bins of the
    instantaneous drive amplitude |H(δL)|, so an amplitude-dependent
    asymmetry (e.g. slip setting in only under compression) shows as the two
    curves diverging with amplitude.

    Returns dict(amp_centers, eta_ext, eta_comp, n_ext, n_comp) or None.
    """
    if res is None:
        res = cfg.get('lin_results', {}).get(f_target)
    if res is None:
        return None

    delta_ref = res['delta_ends']
    delta_xl = res['delta_xl']
    amp = res['amp_env']
    ramp = res['ramp_mask']

    floor = floor_frac * np.abs(delta_ref).max()
    eta_t = delta_xl / np.where(np.abs(delta_ref) > floor, delta_ref, np.nan)

    edges = np.linspace(amp[ramp].min(), amp[ramp].max(), n_amp_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    eta_ext = np.full(n_amp_bins, np.nan)
    eta_comp = np.full(n_amp_bins, np.nan)
    n_ext = np.zeros(n_amp_bins, dtype=int)
    n_comp = np.zeros(n_amp_bins, dtype=int)
    for i in range(n_amp_bins):
        in_bin = ramp & (amp >= edges[i]) & (amp < edges[i + 1])
        e = in_bin & (delta_ref > floor)
        c = in_bin & (delta_ref < -floor)
        n_ext[i], n_comp[i] = int(e.sum()), int(c.sum())
        if e.any():
            eta_ext[i] = float(np.nanmedian(eta_t[e]))
        if c.any():
            eta_comp[i] = float(np.nanmedian(eta_t[c]))
    return dict(amp_centers=centers, eta_ext=eta_ext, eta_comp=eta_comp,
                n_ext=n_ext, n_comp=n_comp)
