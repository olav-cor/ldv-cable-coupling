"""Physical displacement / strain amplitudes across the swept band.

Where the coupling notebooks report *ratios* (η, Θ, transfer functions), this
module reports the raw *amplitudes* that go into them, so they can serve as
reference values for numerical simulations and to spot behavioural differences
between configurations.

Everything is derived from the same windowed bandpass → integrate pipeline as
``analysis.per_frequency_qc`` (so amplitudes are consistent with the η / QC
notebooks). For one target frequency ``f`` the sweep is windowed to
±``n_cycles`` cycles around the log-sweep arrival time
``config.sweep_time_of_frequency(f)``, each velocity channel is bandpassed
(``config.BANDWIDTH_ABS_HZ`` / ``BANDWIDTH_FRAC``) and integrated to displacement
(``signal.integrate_fft``). From the windowed displacement field we read:

    disp_ends      Δu_ends = u∥(shaker end) − u∥(clamp end)     [m]
                   relative axial displacement of the two endpoint sensors that
                   span the hanging segment (this differential is what strains
                   the cable — the shaker's own amplitude is *not* used here).
    strain_ends    ε_ends = Δu_ends / L_ends                     [m/m]
                   segment strain = relative endpoint displacement / spacing.
    disp_shaker    δL = chord-projected shaker displacement       [m]
                   the imposed ground motion (from the *_SHAKER.mat channel).
    disp_first     u∥ at the cable sensor adjacent to the shaker  [m]
                   how much of the drive reaches the first point on the cable.
    transfer_first disp_first / disp_shaker                       [—]
                   shaker → first-sensor displacement transmission (energy ∝ T²).

Two extraction methods are available, selected by ``amp_method``:

* ``'median'`` / ``'peak'`` / ``'rms'`` — **time-domain**. Bandpass the full
  record, window, integrate, and reduce the windowed trace to a single amplitude
  via its analytic (Hilbert) envelope (``_envelope_amplitude``) — the same
  instantaneous amplitude used elsewhere (e.g. ``linearity_qc``'s ``amp_env``).
  Exact for an isolated tone, but the per-frequency full-record bandpass is slow.

* ``'fd'`` — **frequency-domain** (``_amplitude_fd``). Integrate the velocities
  to displacement *once* per dataset, then at each frequency read the amplitude
  straight from a Hann-windowed single-bin FFT at the sweep-arrival time (the
  ``freqdomain.stft_point_transfer`` mechanism, returning an absolute amplitude
  instead of a transfer ratio). No per-frequency bandpass → much faster and
  cheap enough to evaluate on a dense frequency grid. Endpoint differences are
  formed in the time domain before the FFT, so phase between the two ends is
  handled correctly. The single-bin read has minor spectral leakage compared to
  the exact Hilbert envelope, so ratios (``transfer_first``) are the most robust
  output; for the coherence-gated strain-transfer FRF see ``freqdomain``.
"""

import numpy as np
from scipy.signal import hilbert

from .config import (TARGET_FREQS, N_CYCLES_PER_WINDOW,
                     BANDWIDTH_FRAC, BANDWIDTH_ABS_HZ,
                     sweep_time_of_frequency)
from .signal import bandpass, integrate_fft
from .geometry import project_onto_chord
from .analysis import per_frequency_qc, _bp_edges


# Fields returned per frequency (used to build aligned sweep arrays).
_AMP_KEYS = ('f', 'disp_ends', 'strain_ends',
             'disp_shaker', 'disp_first', 'transfer_first', 'L_ends')


def _envelope_amplitude(sig, method='median'):
    """Representative single-tone amplitude of a windowed, bandpassed trace.

    Uses the analytic (Hilbert) envelope |H(sig)|, i.e. the instantaneous
    amplitude of the (nearly mono-frequency) windowed signal.

    method='median' (default) : np.median(|H|) — robust to the bandpass
                                edge-ringing that droops the envelope at the two
                                ends of the ±n_cycles window.
    method='peak'             : |H|.max()      — the largest instantaneous value.
    method='rms'              : sqrt(2)·RMS(sig) — ideal-sinusoid peak.
    """
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        return np.nan
    if method == 'rms':
        return float(np.sqrt(2.0) * np.sqrt(np.mean(sig ** 2)))
    env = np.abs(hilbert(sig))
    if method == 'peak':
        return float(env.max())
    return float(np.median(env))


def _first_sensor_index(cfg):
    """Index of the cable sensor adjacent to the shaker."""
    return cfg['idx_right'] if cfg.get('shaker_end', 'right') == 'right' else cfg['idx_left']


def _window_displacements(cfg, f_target, n_cycles=N_CYCLES_PER_WINDOW,
                          bw_frac=BANDWIDTH_FRAC, bw_abs_hz=BANDWIDTH_ABS_HZ):
    """Window the sweep at f_target and integrate to displacement (light path).

    Mirrors the first three steps of ``analysis.per_frequency_qc`` exactly
    (window → bandpass → integrate → chord projection) but skips the three strain
    methods and η bookkeeping, so a pure-amplitude sweep is roughly twice as fast.

    Returns (u_axial (N_t, N_s), delta_L (N_t,)) or None if the window is too
    short. delta_L carries per_frequency_qc's sign convention (positive = stretch).
    Requires geometry prepared (cfg['chord_unit'], cfg['shaker_end']).
    """
    fs, dt, t = cfg['fs'], cfg['dt'], cfg['t']
    t_c = sweep_time_of_frequency(f_target)
    half = n_cycles / f_target
    mask = (t >= max(t[0], t_c - half)) & (t <= min(t[-1], t_c + half))
    if int(mask.sum()) < 32:
        return None

    f_lo, f_hi = _bp_edges(f_target, bw_frac, bw_abs_hz)
    ux = integrate_fft(bandpass(cfg['vx'], fs, f_lo, f_hi)[mask], dt)
    uy = integrate_fft(bandpass(cfg['vy'], fs, f_lo, f_hi)[mask], dt)
    uz = integrate_fft(bandpass(cfg['vz'], fs, f_lo, f_hi)[mask], dt)
    sh_ux = integrate_fft(bandpass(cfg['sh_vx'], fs, f_lo, f_hi)[mask], dt)
    sh_uy = integrate_fft(bandpass(cfg['sh_vy'], fs, f_lo, f_hi)[mask], dt)
    sh_uz = integrate_fft(bandpass(cfg['sh_vz'], fs, f_lo, f_hi)[mask], dt)

    u_axial = project_onto_chord(ux, uy, uz, cfg['chord_unit'])
    delta_L = project_onto_chord(sh_ux, sh_uy, sh_uz, cfg['chord_unit'])
    if cfg.get('shaker_end') == 'right':
        delta_L = -delta_L
    return u_axial, delta_L


def _fd_axial_signals(cfg):
    """Chord-projected axial displacement of the sensors the FD reader needs.

    Integrates only the three sensor columns actually used (idx_left, idx_right,
    the shaker-side sensor) plus the shaker channel to displacement — done once
    per dataset and cached in ``cfg['_amp_fd_sig']``. This is the single up-front
    cost of the frequency-domain method; every per-frequency read afterwards is a
    cheap windowed FFT.
    """
    cached = cfg.get('_amp_fd_sig')
    if cached is not None:
        return cached

    dt, e = cfg['dt'], cfg['chord_unit']
    il, ir = cfg['idx_left'], cfg['idx_right']
    first = _first_sensor_index(cfg)
    cols = sorted({il, ir, first})

    ux = integrate_fft(cfg['vx'][:, cols], dt)
    uy = integrate_fft(cfg['vy'][:, cols], dt)
    uz = integrate_fft(cfg['vz'][:, cols], dt)
    u_axial = project_onto_chord(ux, uy, uz, e)          # (N_t, len(cols))
    axial = {c: u_axial[:, k] for k, c in enumerate(cols)}

    delta_L = project_onto_chord(integrate_fft(cfg['sh_vx'], dt),
                                 integrate_fft(cfg['sh_vy'], dt),
                                 integrate_fft(cfg['sh_vz'], dt), e)
    if cfg.get('shaker_end') == 'right':
        delta_L = -delta_L

    out = dict(axial=axial, delta_L=delta_L, il=il, ir=ir, first=first,
               t=cfg['t'], fs=cfg['fs'],
               L_ends=float(np.linalg.norm(cfg['XYZ'][ir] - cfg['XYZ'][il])))
    cfg['_amp_fd_sig'] = out
    return out


def _amplitude_fd(cfg, f_target, n_cycles=N_CYCLES_PER_WINDOW,
                  bw_frac=BANDWIDTH_FRAC, bw_abs_hz=BANDWIDTH_ABS_HZ):
    """Frequency-domain amplitudes at one swept frequency (band-energy FFT).

    Windows the once-integrated displacement signals at the sweep-arrival time,
    Hann-tapers, and recovers the tone amplitude from the FFT **energy in the
    band** [f_lo, f_hi] = the same ±``BANDWIDTH_ABS_HZ`` band the time-domain
    path bandpasses. Using the band energy (Parseval) instead of a single bin is
    essential at low frequency, where the ±``n_cycles``-cycle window is long in
    time and the log-sweep smears the tone across many bins.

    Amplitude of a tone of peak A windowed by w satisfies (Parseval, single
    tone in-band):  A = 2·sqrt( Σ_band |X_k|² / (N·Σ w²) ). Returns the same
    keys as ``amplitude_at_frequency``.
    """
    sig = _fd_axial_signals(cfg)
    t, fs = sig['t'], sig['fs']
    t_c = sweep_time_of_frequency(f_target)
    half = n_cycles / f_target
    mask = (t >= max(t[0], t_c - half)) & (t <= min(t[-1], t_c + half))
    n = int(mask.sum())
    if n < 32:
        return None

    win = np.hanning(n)
    w2N = float(np.sum(win ** 2)) * n                     # N · Σw²
    bins = np.fft.rfftfreq(n, d=1.0 / fs)
    f_lo, f_hi = _bp_edges(f_target, bw_frac, bw_abs_hz)
    band = (bins >= f_lo) & (bins <= f_hi)
    if not band.any():                                   # band narrower than a bin
        band = np.zeros(bins.shape, dtype=bool)
        band[int(np.argmin(np.abs(bins - f_target)))] = True

    def _amp(x1d):
        X = np.fft.rfft(x1d[mask] * win)
        return 2.0 * float(np.sqrt(np.sum(np.abs(X[band]) ** 2) / w2N))

    ax = sig['axial']
    L_ends = sig['L_ends']
    disp_ends = _amp(ax[sig['ir']] - ax[sig['il']])
    disp_shaker = _amp(sig['delta_L'])
    disp_first = _amp(ax[sig['first']])
    transfer_first = disp_first / disp_shaker if disp_shaker > 0 else np.nan

    return dict(
        f=float(f_target),
        disp_ends=disp_ends,
        strain_ends=disp_ends / L_ends,
        disp_shaker=disp_shaker,
        disp_first=disp_first,
        transfer_first=transfer_first,
        L_ends=L_ends,
    )


def amplitude_at_frequency(cfg, f_target, amp_method='median', use_cache=True,
                           n_cycles=N_CYCLES_PER_WINDOW,
                           bw_frac=BANDWIDTH_FRAC, bw_abs_hz=BANDWIDTH_ABS_HZ):
    """Endpoint / shaker displacement & strain amplitudes at one swept frequency.

    ``amp_method='fd'`` uses the fast frequency-domain single-bin reader
    (``_amplitude_fd``). Otherwise the time-domain path is used: if
    ``cfg['per_freq'][f_target]`` already exists (populated by
    ``per_frequency_qc`` in the η / QC notebooks) it is reused for free,
    else the light ``_window_displacements`` path is taken and the windowed
    trace is reduced with ``_envelope_amplitude(..., amp_method)``.

    Returns a dict with the keys in ``_AMP_KEYS`` (all scalars, SI units), or
    None if the window is too short.
    """
    if amp_method == 'fd':
        return _amplitude_fd(cfg, f_target, n_cycles=n_cycles)

    il, ir = cfg['idx_left'], cfg['idx_right']
    first = _first_sensor_index(cfg)

    res = cfg.get('per_freq', {}).get(f_target) if use_cache else None
    if res is not None:
        u_axial = project_onto_chord(res['ux'], res['uy'], res['uz'], cfg['chord_unit'])
        delta_L = res['delta_L_win']
        delta_ends = res['delta_ends']
        eps_ends = res['eps_ref_ends']
        L_ends = res['L_ends']
    else:
        out = _window_displacements(cfg, f_target, n_cycles=n_cycles,
                                    bw_frac=bw_frac, bw_abs_hz=bw_abs_hz)
        if out is None:
            return None
        u_axial, delta_L = out
        L_ends = float(np.linalg.norm(cfg['XYZ'][ir] - cfg['XYZ'][il]))
        delta_ends = u_axial[:, ir] - u_axial[:, il]
        eps_ends = delta_ends / L_ends

    disp_shaker = _envelope_amplitude(delta_L, amp_method)
    disp_first = _envelope_amplitude(u_axial[:, first], amp_method)
    transfer_first = disp_first / disp_shaker if disp_shaker > 0 else np.nan

    return dict(
        f=float(f_target),
        disp_ends=_envelope_amplitude(delta_ends, amp_method),
        strain_ends=_envelope_amplitude(eps_ends, amp_method),
        disp_shaker=disp_shaker,
        disp_first=disp_first,
        transfer_first=transfer_first,
        L_ends=L_ends,
    )


def amplitude_sweep(cfg, freqs=None, amp_method='median', use_cache=True,
                    verbose=False, **bpkw):
    """Run ``amplitude_at_frequency`` over a frequency grid.

    Returns a dict of 1-D arrays (keys in ``_AMP_KEYS``), aligned by index;
    frequencies whose window was too short are skipped. The result is cached in
    ``cfg['amp_sweep'][amp_method]`` so repeated plots are free.

    freqs : iterable of target frequencies [Hz]; default ``config.TARGET_FREQS``.
    """
    if freqs is None:
        freqs = TARGET_FREQS
    cached = cfg.get('amp_sweep', {}).get(amp_method)
    if cached is not None and use_cache and len(cached['f']) == len(list(freqs)):
        return cached

    acc = {k: [] for k in _AMP_KEYS}
    for f in freqs:
        d = amplitude_at_frequency(cfg, f, amp_method=amp_method,
                                   use_cache=use_cache, **bpkw)
        if d is None:
            continue
        for k in _AMP_KEYS:
            acc[k].append(d[k])
        if verbose:
            print(f"  {f:5.1f} Hz  Δu_ends={d['disp_ends']*1e6:8.3f} µm  "
                  f"ε_ends={d['strain_ends']*1e6:8.2f} µε  "
                  f"T(sh→1)={d['transfer_first']:.3f}")
    out = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
    cfg.setdefault('amp_sweep', {})[amp_method] = out
    return out


# Quantities that carry a length dimension (metres) → scaled to µm for display.
_DISP_QUANTITIES = ('disp_ends', 'disp_shaker', 'disp_first')
_STRAIN_QUANTITIES = ('strain_ends',)


def band_mean_amplitude(cfg, f_min=0.0, f_max=25.0, amp_method='median',
                        freqs=None, use_cache=True, **bpkw):
    """Mean (± std) of each amplitude over the frequency band [f_min, f_max].

    Reuses ``cfg['amp_sweep'][amp_method]`` when present. Returns a dict with
    ``<quantity>_mean`` / ``<quantity>_std`` for every quantity plus ``n`` (the
    number of frequencies averaged) and the band limits. The default band
    (0–25 Hz) is the quasi-static range requested for the configuration overview.
    """
    sw = cfg.get('amp_sweep', {}).get(amp_method)
    if sw is None:
        sw = amplitude_sweep(cfg, freqs=freqs, amp_method=amp_method,
                             use_cache=use_cache, **bpkw)
    m = (sw['f'] >= f_min) & (sw['f'] <= f_max)
    out = dict(n=int(m.sum()), f_min=float(f_min), f_max=float(f_max))
    for k in ('disp_ends', 'strain_ends', 'disp_shaker',
              'disp_first', 'transfer_first'):
        vals = sw[k][m]
        vals = vals[np.isfinite(vals)]
        out[k + '_mean'] = float(np.mean(vals)) if vals.size else np.nan
        out[k + '_std'] = float(np.std(vals)) if vals.size else np.nan
    return out
