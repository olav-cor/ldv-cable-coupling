"""Per-frequency QC analysis pipeline."""

import numpy as np
from scipy.signal import hilbert

from .config import (N_CYCLES_PER_WINDOW, BANDWIDTH_FRAC, TARGET_FREQS,
                     sweep_time_of_frequency,
                     LIN_FREQUENCIES, LIN_SEGMENT_DURATION, LIN_FADE_DURATION,
                     linearity_segment_window,
                     theta_from_sag, theta_from_material)
from .signal import bandpass, integrate_fft, amplitude_spectrum, find_dominant_peaks
from .geometry import project_onto_chord
from .strain import strain_spatial_gradient, strain_3d_arclength


def spectral_peak_summary(cfg, components=('vx', 'vy', 'vz'),
                          f_max=500.0, n_peaks=4):
    """Compute the mean amplitude spectrum and find dominant peaks per component.

    Works for cable cfgs (2-D arrays, mean taken across sensors) and shaker-ref
    dicts (1-D arrays).

    Parameters
    ----------
    cfg        : loaded cfg dict or shaker-ref dict.
    components : component keys present in cfg (e.g. 'vx' or 'sh_vx').
    f_max      : upper frequency limit for peak search [Hz].
    n_peaks    : number of dominant peaks to return per component.

    Returns
    -------
    dict keyed by component name, each value a dict with:
        freqs     : 1-D array  frequency bins [Hz]
        mean_spec : 1-D array  mean spectrum [same units as input × 2/N]
        peaks     : list of (freq_hz, amp_normalized) sorted by amp desc.
    """
    result = {}
    for comp in components:
        arr = np.asarray(cfg[comp])
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        freqs, spec = amplitude_spectrum(arr, cfg['fs'])
        mean_spec = spec.mean(axis=1)
        peaks = find_dominant_peaks(freqs, mean_spec, n_peaks=n_peaks, f_max=f_max)
        result[comp] = {'freqs': freqs, 'mean_spec': mean_spec, 'peaks': peaks}
    return result


def per_frequency_qc(cfg, f_target, n_cycles=N_CYCLES_PER_WINDOW,
                     bw_frac=BANDWIDTH_FRAC, grad_direction="chord"):
    """Complete single-frequency QC for one dataset.

    Steps:
        1. Time window centred on the sweep arrival at f_target.
        2. Bandpass cable + shaker velocities around f_target.
        3. Integrate to displacement.
        4. Strain via Method 1 (spatial gradient) and Method 2 (arc-length).
        5. Cable elongation δxₗ vs shaker δL → η(t), median η over window.

    Returns a dict of windowed arrays + summary scalars, or None if the window
    is too short.
    """
    fs, dt, t = cfg['fs'], cfg['dt'], cfg['t']

    # 1) Time window
    t_c = sweep_time_of_frequency(f_target)
    half = n_cycles / f_target
    t_lo, t_hi = max(t[0], t_c - half), min(t[-1], t_c + half)
    mask = (t >= t_lo) & (t <= t_hi)
    t_win = t[mask]
    if len(t_win) < 32:
        print(f"  [{f_target} Hz] window too short - skipping.")
        return None

    # 2) Bandpass
    f_lo, f_hi = f_target * (1 - bw_frac), f_target * (1 + bw_frac)
    vx_bp = bandpass(cfg['vx'], fs, f_lo, f_hi)[mask]
    vy_bp = bandpass(cfg['vy'], fs, f_lo, f_hi)[mask]
    vz_bp = bandpass(cfg['vz'], fs, f_lo, f_hi)[mask]
    sh_vx_bp = bandpass(cfg['sh_vx'], fs, f_lo, f_hi)[mask]
    sh_vy_bp = bandpass(cfg['sh_vy'], fs, f_lo, f_hi)[mask]
    sh_vz_bp = bandpass(cfg['sh_vz'], fs, f_lo, f_hi)[mask]

    # 3) Integrate → displacement
    ux = integrate_fft(vx_bp, dt)
    uy = integrate_fft(vy_bp, dt)
    uz = integrate_fft(vz_bp, dt)
    sh_ux = integrate_fft(sh_vx_bp, dt)
    sh_uy = integrate_fft(sh_vy_bp, dt)
    sh_uz = integrate_fft(sh_vz_bp, dt)
    delta_L_win = project_onto_chord(sh_ux, sh_uy, sh_uz, cfg['chord_unit'])
    if cfg['shaker_end'] == 'right':
        delta_L_win = -delta_L_win

    # 4) Strain — Method 1 (spatial gradient)
    strain_grad = strain_spatial_gradient(
        cfg['XYZ'], ux, uy, uz, cfg['chord_unit'], direction=grad_direction)

    # Integrate Method 1 over the gap to get a total elongation comparable to M2
    s_coords = np.array([np.dot(cfg['XYZ'][i] - cfg['XYZ'][0], cfg['chord_unit'])
                         for i in range(len(cfg['XYZ']))])
    s_gap = s_coords[cfg['idx_left']:cfg['idx_right'] + 1]
    sg_gap = strain_grad[:, cfg['idx_left']:cfg['idx_right'] + 1]
    delta_xl_m1 = np.trapz(np.where(np.isnan(sg_gap), 0.0, sg_gap), s_gap, axis=1)

    # 4) Strain — Method 2 (arc-length per segment)
    delta_d, L0_seg, seg_strain, delta_xl = strain_3d_arclength(
        cfg['XYZ'], ux, uy, uz, cfg['idx_left'], cfg['idx_right'])

    # End-sensor reference (computed from the same windowed/integrated data
    # so the time axis matches t_win exactly).
    xyz_left = cfg['XYZ'][cfg['idx_left']]
    xyz_right = cfg['XYZ'][cfg['idx_right']]
    L_ends = np.linalg.norm(xyz_right - xyz_left)
    ux_ends = ux[:, [cfg['idx_left'], cfg['idx_right']]]
    uy_ends = uy[:, [cfg['idx_left'], cfg['idx_right']]]
    uz_ends = uz[:, [cfg['idx_left'], cfg['idx_right']]]
    u_axial_ends = project_onto_chord(ux_ends, uy_ends, uz_ends, cfg['chord_unit'])
    eps_ref_ends = (u_axial_ends[:, 1] - u_axial_ends[:, 0]) / L_ends

    delta_ends = u_axial_ends[:, 1] - u_axial_ends[:, 0]
    ends_floor = 0.05 * np.abs(delta_ends).max()
    dL_floor = 0.05 * np.abs(delta_L_win).max()

    # Shaker-to-end reference: relative displacement between the shaker and
    # the cable endpoint sensor adjacent to the shaker.
    # Sign: after delta_L_win's sign correction, subtracting the adjacent
    # sensor's axial displacement gives the elongation of the sub-gap between
    # the shaker and the first cable sensor.
    L_shend = cfg['chord_len']
    if cfg['shaker_end'] == 'right':
        delta_shend = delta_L_win + u_axial_ends[:, 1]
    else:  # 'left'
        delta_shend = delta_L_win - u_axial_ends[:, 0]
    shend_floor = 0.05 * np.abs(delta_shend).max()
    eps_ref_shend = delta_shend / L_shend

    # 5) Coupling efficiency — Method 2 (arc-length)
    eta_t = delta_xl / np.where(np.abs(delta_L_win) > dL_floor, delta_L_win, np.nan)
    eta_med = float(np.nanmedian(eta_t))
    eta_t_ends = delta_xl / np.where(np.abs(delta_ends) > ends_floor, delta_ends, np.nan)
    eta_med_ends = float(np.nanmedian(eta_t_ends))
    eta_t_shend = delta_xl / np.where(np.abs(delta_shend) > shend_floor, delta_shend, np.nan)
    eta_med_shend = float(np.nanmedian(eta_t_shend))

    # 5) Coupling efficiency — Method 1 (spatial gradient integrated)
    eta_t_m1 = delta_xl_m1 / np.where(np.abs(delta_L_win) > dL_floor, delta_L_win, np.nan)
    eta_med_m1 = float(np.nanmedian(eta_t_m1))
    eta_t_m1_ends = delta_xl_m1 / np.where(np.abs(delta_ends) > ends_floor, delta_ends, np.nan)
    eta_med_m1_ends = float(np.nanmedian(eta_t_m1_ends))
    eta_t_m1_shend = delta_xl_m1 / np.where(np.abs(delta_shend) > shend_floor, delta_shend, np.nan)
    eta_med_m1_shend = float(np.nanmedian(eta_t_m1_shend))

    # 5b) Envelope-ratio estimator (phase-insensitive): |H(δxₗ)| / |H(δref)|
    # The Hilbert envelope gives the instantaneous amplitude of the bandpassed
    # signal, so the ratio is independent of the phase shift between cable and
    # shaker/reference — useful for diagnosing whether low η is amplitude- or
    # phase-driven.
    env_xl   = np.abs(hilbert(delta_xl))
    env_xl_m1 = np.abs(hilbert(delta_xl_m1))
    env_L    = np.abs(hilbert(delta_L_win))
    env_ends = np.abs(hilbert(delta_ends))
    env_shend = np.abs(hilbert(delta_shend))

    eta_t_env       = env_xl    / np.where(env_L     > dL_floor,    env_L,     np.nan)
    eta_med_env      = float(np.nanmedian(eta_t_env))
    eta_t_env_ends  = env_xl    / np.where(env_ends  > ends_floor,  env_ends,  np.nan)
    eta_med_env_ends = float(np.nanmedian(eta_t_env_ends))
    eta_t_env_shend = env_xl    / np.where(env_shend > shend_floor, env_shend, np.nan)
    eta_med_env_shend = float(np.nanmedian(eta_t_env_shend))

    eta_t_m1_env       = env_xl_m1 / np.where(env_L     > dL_floor,    env_L,     np.nan)
    eta_med_m1_env      = float(np.nanmedian(eta_t_m1_env))
    eta_t_m1_env_ends  = env_xl_m1 / np.where(env_ends  > ends_floor,  env_ends,  np.nan)
    eta_med_m1_env_ends = float(np.nanmedian(eta_t_m1_env_ends))
    eta_t_m1_env_shend = env_xl_m1 / np.where(env_shend > shend_floor, env_shend, np.nan)
    eta_med_m1_env_shend = float(np.nanmedian(eta_t_m1_env_shend))

    return dict(
        f_target=f_target, t_win=t_win,
        vx=vx_bp, vy=vy_bp, vz=vz_bp,
        ux=ux, uy=uy, uz=uz,
        delta_L_win=delta_L_win,
        strain_grad=strain_grad,
        seg_strain=seg_strain, delta_d=delta_d, L0_seg=L0_seg,
        delta_xl=delta_xl, delta_xl_m1=delta_xl_m1,
        eps_ref_ends=eps_ref_ends, L_ends=L_ends,
        delta_ends=delta_ends,
        delta_shend=delta_shend, L_shend=L_shend, eps_ref_shend=eps_ref_shend,
        eta_t=eta_t, eta_med=eta_med,
        eta_t_ends=eta_t_ends, eta_med_ends=eta_med_ends,
        eta_t_shend=eta_t_shend, eta_med_shend=eta_med_shend,
        eta_t_m1=eta_t_m1, eta_med_m1=eta_med_m1,
        eta_t_m1_ends=eta_t_m1_ends, eta_med_m1_ends=eta_med_m1_ends,
        eta_t_m1_shend=eta_t_m1_shend, eta_med_m1_shend=eta_med_m1_shend,
        eta_t_env=eta_t_env, eta_med_env=eta_med_env,
        eta_t_env_ends=eta_t_env_ends, eta_med_env_ends=eta_med_env_ends,
        eta_t_env_shend=eta_t_env_shend, eta_med_env_shend=eta_med_env_shend,
        eta_t_m1_env=eta_t_m1_env, eta_med_m1_env=eta_med_m1_env,
        eta_t_m1_env_ends=eta_t_m1_env_ends, eta_med_m1_env_ends=eta_med_m1_env_ends,
        eta_t_m1_env_shend=eta_t_m1_env_shend, eta_med_m1_env_shend=eta_med_m1_env_shend,
        f_lo=f_lo, f_hi=f_hi,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Linearity-test analysis
# ─────────────────────────────────────────────────────────────────────────────

def prepare_geometry(cfg, sag_use_3d=True, sag_use_parabola=False):
    """Compute chord geometry and annotate cfg in-place (no plots).

    Sets chord_unit, chord_len, chord_start, static_sag, idx_sag,
    theta_pred, eta_pred. Call this before linearity_qc or per_frequency_qc
    whenever plot_initial_geometry has not been run.

    Parameters
    ----------
    sag_use_3d      : passed to measure_sag — True = full 3-D perpendicular
                      distance (default), False = x-z plane projection.
    sag_use_parabola: passed to measure_sag — True = fit a parabola through
                      the cable points before finding the peak sag.
    """
    from .geometry import compute_chord, measure_sag
    e_hat, L_chord, P1 = compute_chord(cfg['XYZ'], cfg['idx_left'], cfg['idx_right'])
    sag, idx_sag, _ = measure_sag(cfg['XYZ'], cfg['idx_left'], cfg['idx_right'],
                                   use_3d=sag_use_3d, use_parabola=sag_use_parabola)
    cable_name = cfg.get('cable')
    # Sag-based Θ: general formula A·w₀²/(8I), valid for any cross-section.
    # For circular cables this equals (1/2)(w₀/r)².
    theta_sag = theta_from_sag(cable_name, sag)
    if theta_sag is None:
        # Fallback for datasets without a CABLE_PROPERTIES entry.
        theta_sag = 0.5 * (sag / cfg['radius_m']) ** 2

    # Material-based Θ: requires ρ and E to be filled in CABLE_PROPERTIES.
    theta_mat = theta_from_material(cable_name, L_chord)

    cfg.update(dict(
        chord_unit=e_hat, chord_len=L_chord, chord_start=P1,
        static_sag=sag, idx_sag=idx_sag,
        theta_pred=theta_sag,
        eta_pred=1.0 / (1.0 + theta_sag),
        theta_material=theta_mat,
        eta_material=(None if theta_mat is None else 1.0 / (1.0 + theta_mat)),
    ))
    return cfg


def linearity_qc(cfg, f_target, bw_frac=BANDWIDTH_FRAC, grad_direction='chord'):
    """Full-segment coupling efficiency for one frequency in the linearity test.

    Unlike per_frequency_qc (which uses a short window centred on the log-sweep
    arrival), this function extracts the *entire* 3.5 s mono-frequency segment
    so the amplitude-vs-η dependence can be tracked over the full ramp.

    Requires cfg to have chord_unit set (call prepare_geometry first).

    Returns a dict with:
        t_win       absolute time axis of the segment [s]
        t_start     absolute start time [s]
        t_ramp_end  time within segment (0-based) where ramp ends [s]
        ramp_mask   boolean (N_t,) — True during the rising ramp
        delta_L_win shaker elongation δL [m], bandpassed
        delta_xl    cable elongation δxₗ [m]
        amp_env     Hilbert envelope of δL [m] — instantaneous amplitude
        eta_t       η(t) = δxₗ / δL sample-by-sample
        eta_med     median η over the whole segment
        eta_t_ends  η(t) referenced to end-sensor elongation
        eta_med_ends median η (end-sensor reference)
        ... plus raw bandpassed vx/vy/vz, ux/uy/uz, strain fields
    """
    from scipy.signal import hilbert

    fs, dt, t = cfg['fs'], cfg['dt'], cfg['t']
    t_start, t_end = linearity_segment_window(f_target)

    mask = (t >= t_start) & (t < t_end)
    t_win = t[mask]

    min_samples = max(32, int(5.0 / f_target * fs))
    if len(t_win) < min_samples:
        print(f"  [{f_target} Hz] segment too short ({len(t_win)} samples) — skipping.")
        return None

    f_lo = max(f_target * (1 - bw_frac), 0.5)
    f_hi = f_target * (1 + bw_frac)

    vx_bp = bandpass(cfg['vx'], fs, f_lo, f_hi)[mask]
    vy_bp = bandpass(cfg['vy'], fs, f_lo, f_hi)[mask]
    vz_bp = bandpass(cfg['vz'], fs, f_lo, f_hi)[mask]
    sh_vx_bp = bandpass(cfg['sh_vx'], fs, f_lo, f_hi)[mask]
    sh_vy_bp = bandpass(cfg['sh_vy'], fs, f_lo, f_hi)[mask]
    sh_vz_bp = bandpass(cfg['sh_vz'], fs, f_lo, f_hi)[mask]

    ux = integrate_fft(vx_bp, dt)
    uy = integrate_fft(vy_bp, dt)
    uz = integrate_fft(vz_bp, dt)
    sh_ux = integrate_fft(sh_vx_bp, dt)
    sh_uy = integrate_fft(sh_vy_bp, dt)
    sh_uz = integrate_fft(sh_vz_bp, dt)

    delta_L_win = project_onto_chord(sh_ux, sh_uy, sh_uz, cfg['chord_unit'])
    if cfg['shaker_end'] == 'right':
        delta_L_win = -delta_L_win

    strain_grad = strain_spatial_gradient(
        cfg['XYZ'], ux, uy, uz, cfg['chord_unit'], direction=grad_direction)

    delta_d, L0_seg, seg_strain, delta_xl = strain_3d_arclength(
        cfg['XYZ'], ux, uy, uz, cfg['idx_left'], cfg['idx_right'])

    # End-sensor reference
    xyz_left = cfg['XYZ'][cfg['idx_left']]
    xyz_right = cfg['XYZ'][cfg['idx_right']]
    L_ends = np.linalg.norm(xyz_right - xyz_left)
    ux_ends = ux[:, [cfg['idx_left'], cfg['idx_right']]]
    uy_ends = uy[:, [cfg['idx_left'], cfg['idx_right']]]
    uz_ends = uz[:, [cfg['idx_left'], cfg['idx_right']]]
    u_axial_ends = project_onto_chord(ux_ends, uy_ends, uz_ends, cfg['chord_unit'])
    delta_ends = u_axial_ends[:, 1] - u_axial_ends[:, 0]

    ends_floor = 0.05 * np.abs(delta_ends).max()
    eta_t_ends = delta_xl / np.where(
        np.abs(delta_ends) > ends_floor, delta_ends, np.nan)
    eta_med_ends = float(np.nanmedian(eta_t_ends))

    dL_floor = 0.05 * np.abs(delta_L_win).max()
    eta_t = delta_xl / np.where(
        np.abs(delta_L_win) > dL_floor, delta_L_win, np.nan)
    eta_med = float(np.nanmedian(eta_t))

    # Shaker-to-end reference
    L_shend = cfg['chord_len']
    if cfg['shaker_end'] == 'right':
        delta_shend = delta_L_win + u_axial_ends[:, 1]
    else:  # 'left'
        delta_shend = delta_L_win - u_axial_ends[:, 0]
    shend_floor = 0.05 * np.abs(delta_shend).max()
    eta_t_shend = delta_xl / np.where(np.abs(delta_shend) > shend_floor, delta_shend, np.nan)
    eta_med_shend = float(np.nanmedian(eta_t_shend))

    # Hilbert envelope of δL — true instantaneous amplitude
    amp_env = np.abs(hilbert(delta_L_win))

    # Ramp/fade boundary: ramp ends (LIN_SEGMENT_DURATION - LIN_FADE_DURATION) into segment
    t_ramp_end = LIN_SEGMENT_DURATION - LIN_FADE_DURATION
    t_rel = t_win - t_start
    ramp_mask = t_rel < t_ramp_end

    return dict(
        f_target=f_target, f_lo=f_lo, f_hi=f_hi,
        t_win=t_win, t_start=t_start, t_end=t_end,
        t_ramp_end=t_ramp_end, ramp_mask=ramp_mask,
        vx=vx_bp, vy=vy_bp, vz=vz_bp,
        ux=ux, uy=uy, uz=uz,
        delta_L_win=delta_L_win,
        delta_xl=delta_xl,
        amp_env=amp_env,
        eta_t=eta_t, eta_med=eta_med,
        eta_t_ends=eta_t_ends, eta_med_ends=eta_med_ends,
        eta_t_shend=eta_t_shend, eta_med_shend=eta_med_shend,
        delta_ends=delta_ends, L_ends=L_ends,
        delta_shend=delta_shend, L_shend=L_shend,
        strain_grad=strain_grad,
        seg_strain=seg_strain, delta_d=delta_d, L0_seg=L0_seg,
    )


def run_linearity_analysis(cfg, frequencies=None, bw_frac=BANDWIDTH_FRAC,
                            grad_direction='chord', verbose=True):
    """Run linearity_qc for every frequency and cache in cfg['lin_results'].

    Requires prepare_geometry (or plot_initial_geometry) to have been run first.

    Returns the results dict (also stored in cfg['lin_results']).
    """
    if frequencies is None:
        frequencies = LIN_FREQUENCIES
    results = {}
    for f in frequencies:
        if verbose:
            print(f"  {f:4.0f} Hz ...", end=' ', flush=True)
        res = linearity_qc(cfg, f, bw_frac=bw_frac, grad_direction=grad_direction)
        if res is not None:
            results[f] = res
            if verbose:
                print(f"η_med = {res['eta_med']:.3f}")
        elif verbose:
            print("skipped")
    cfg['lin_results'] = results
    return results


def compute_mean_eta(cfg, f_min=0.0, f_max=50.0, ref='shaker',
                     bw_frac=BANDWIDTH_FRAC, grad_direction='chord',
                     strain_method='arclength', estimator='instantaneous',
                     verbose=False):
    """Compute mean coupling efficiency over the frequency range [f_min, f_max].

    Selects all TARGET_FREQS within [f_min, f_max]. For each, reuses
    cfg['per_freq'][f] if already computed (from another notebook or a prior
    call); otherwise runs per_frequency_qc on demand.

    Requires cfg['chord_unit'] to be set (call prepare_geometry first).

    Parameters
    ----------
    cfg           : loaded cable cfg dict with geometry already computed.
    f_min, f_max  : frequency band [Hz] over which η is averaged.
    ref           : 'shaker' or 'ends' — which reference elongation to use.
    bw_frac       : bandpass half-width fraction (only for newly computed freqs).
    grad_direction: 'chord' or 'cartesian' (only for newly computed freqs).
    strain_method : 'arclength' — Σδd_i per segment (Method 2, default)
                    'gradient'  — ∫ε ds spatial-gradient integrated (Method 1)
    estimator     : 'instantaneous' (default) — η = δxₗ(t) / δref(t), median
                    'envelope'                — η = |H(δxₗ)| / |H(δref)|, median
                    The envelope estimator is phase-insensitive: it measures the
                    amplitude ratio of the two signals regardless of any phase
                    shift between them.
    verbose       : print progress.

    Returns
    -------
    (eta_mean, eta_std, n_freqs) — or (None, None, 0) if no valid results.
    Also stores in cfg['eta_summary'][(f_min, f_max, ref, strain_method, estimator)].
    """
    target_fs = [f for f in TARGET_FREQS if f_min <= f <= f_max]
    if not target_fs:
        print(f"  [{cfg.get('label', '?')}] No TARGET_FREQS in [{f_min}, {f_max}] Hz.")
        return None, None, 0

    if 'chord_unit' not in cfg:
        print(f"  [{cfg.get('label', '?')}] Geometry not computed — call prepare_geometry first.")
        return None, None, 0

    results = cfg.setdefault('per_freq', {})
    _env = estimator == 'envelope'
    if strain_method == 'gradient':
        if ref == 'ends':
            eta_key = 'eta_med_m1_env_ends' if _env else 'eta_med_m1_ends'
        elif ref == 'shaker_end':
            eta_key = 'eta_med_m1_env_shend' if _env else 'eta_med_m1_shend'
        else:
            eta_key = 'eta_med_m1_env' if _env else 'eta_med_m1'
    else:
        if ref == 'ends':
            eta_key = 'eta_med_env_ends' if _env else 'eta_med_ends'
        elif ref == 'shaker_end':
            eta_key = 'eta_med_env_shend' if _env else 'eta_med_shend'
        else:
            eta_key = 'eta_med_env' if _env else 'eta_med'

    for f in target_fs:
        # Recompute if not cached, or if cached result is missing the requested key
        # (e.g. after a code update that added a new eta variant).
        if f not in results or eta_key not in results[f]:
            if verbose:
                print(f"  {f:.0f} Hz ...", end=' ', flush=True)
            res = per_frequency_qc(cfg, f, bw_frac=bw_frac,
                                    grad_direction=grad_direction)
            if res is not None:
                results[f] = res
            if verbose:
                print("ok" if res is not None else "skip")

    freqs_ok = [f for f in target_fs if f in results]
    etas = [results[f][eta_key] for f in freqs_ok
            if eta_key in results[f]]
    if not etas:
        return None, None, 0

    eta_mean = float(np.mean(etas))
    eta_std = float(np.std(etas))

    summary = cfg.setdefault('eta_summary', {})
    summary[(f_min, f_max, ref, strain_method, estimator)] = {
        'eta_mean': eta_mean,
        'eta_std': eta_std,
        'n_freqs': len(etas),
        'freqs_used': freqs_ok,
    }
    return eta_mean, eta_std, len(etas)
