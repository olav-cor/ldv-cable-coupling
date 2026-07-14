"""Figures and the interactive browser for the modal-analysis pipeline.

All functions consume the results produced by
:func:`ldv_analysis.modal.analyze_dataset` (stored in ``cfg['modal']``); none
of them recompute the analysis.

Per-dataset deep dives (§6):
    plot_fingerprint        — spectral fingerprint with marked resonances
    plot_mode_panels        — 6-panel φ_y/φ_z magnitude + phase + theory + MAC
    plot_mac_heatmap        — measured × theoretical MAC matrix
    plot_ellipse_map        — polarization ellipses along the cable, one row/mode
    plot_phase_snapshots    — 3-D cable shape at 0/90/180/270° (call on demand)
    deep_dive               — convenience: fingerprint + every mode's panels

Cross-dataset summaries (§7):
    plot_mode_frequency_chart
    plot_fn_vs_gap
    build_summary_table

Interactive (§8):
    modal_dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

from .config import dataset_style, CABLE_COLORS
from .modal import (theoretical_modes, ellipse_trace, classify_polarization,
                    analyze_dataset)


# Marker per polarization class — used across the cross-dataset summaries.
POL_MARKERS = {'linear': 'o', 'elliptical': 's', 'circular': '*'}
POL_COLORS = {'linear': '#1f77b4', 'elliptical': '#ff7f0e', 'circular': '#d62728'}


def _require_modal(cfg):
    if 'modal' not in cfg:
        raise RuntimeError(
            f"cfg['{cfg.get('label', '?')}'] has no modal results — "
            "run modal.analyze_dataset(cfg) first.")
    return cfg['modal']


# ─────────────────────────────────────────────────────────────────────────────
# §6  Spectral fingerprint
# ─────────────────────────────────────────────────────────────────────────────
def plot_fingerprint(cfg, ax=None, show_total=True):
    """Plot the v_y and v_z fingerprints (linear y) with detected resonances marked.

    Solid lines are the per-polarization max-amplitude fingerprints; the faint
    grey curve is the combined total-power detection signal (scaled). Vertical
    dashed lines + labels mark the detected resonances.
    """
    md = _require_modal(cfg)
    freqs = md['freqs']
    p = md['params']
    band = (freqs >= p['f_min']) & (freqs <= p['f_max'])

    if ax is None:
        _fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(freqs[band], md['fp_y_max'][band] * 1e3, lw=1.0,
            color='#0022FF', label=r'$\max_s |V_y|$')
    ax.plot(freqs[band], md['fp_z_max'][band] * 1e3, lw=1.0,
            color='#E600FF', label=r'$\max_s |V_z|$')

    if show_total:
        # total-power detection signal scaled to the visible fingerprint range
        ymax = max(md['fp_y_max'][band].max(), md['fp_z_max'][band].max()) * 1e3
        tot = md['sig_norm'][band] * ymax
        ax.fill_between(freqs[band], 0, tot, color='gray', alpha=0.15,
                        label='total power (norm.)')

    y_top = ax.get_ylim()[1]
    for f_n in md['peaks']:
        ax.axvline(f_n, color='k', ls='--', lw=0.7, alpha=0.55)
        ax.text(f_n, y_top * 0.98, f'{f_n:.0f}', rotation=90, va='top',
                ha='right', fontsize=8, color='k')

    ax.set_xlim(p['f_min'], p['f_max'])
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [mm/s]')
    ax.set_title(f"{cfg['label']} — spectral fingerprint "
                 f"({len(md['peaks'])} resonances)")
    ax.legend(fontsize=8, loc='upper right')
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# §6  Per-resonance mode-shape panels
# ─────────────────────────────────────────────────────────────────────────────
def plot_mode_panels(cfg, mode_index):
    """6-panel deep dive for one resonance.

    (a) |φ_y|  (b) |φ_z|  (c) ∠φ_y  (d) ∠φ_z
    (e) measured dominant magnitude vs best-matching theoretical mode
    (f) text panel: MAC, mode order, polarization summary.
    """
    md = _require_modal(cfg)
    mode = md['modes'][mode_index]
    s_mm = md['s_mm']
    col, _, _ = dataset_style(cfg)

    phi_y, phi_z = mode['phi_y'], mode['phi_z']
    ay, az = np.abs(phi_y), np.abs(phi_z)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.5))
    (axa, axb, axe), (axc, axd, axf) = axes

    axa.plot(s_mm, ay * 1e6, 'o-', color='#0022FF', ms=4)
    axa.set_title(r'(a) $|\varphi_y|$'); axa.set_ylabel('|disp.| [µm]')
    axb.plot(s_mm, az * 1e6, 'o-', color='#E600FF', ms=4)
    axb.set_title(r'(b) $|\varphi_z|$')

    axc.plot(s_mm, np.degrees(np.angle(phi_y)), 'o-', color='#0022FF', ms=4)
    axc.set_title(r'(c) $\angle\varphi_y$'); axc.set_ylabel('phase [°]')
    axc.set_ylim(-185, 185); axc.set_xlabel('Position along gap [mm]')
    axd.plot(s_mm, np.degrees(np.angle(phi_z)), 'o-', color='#E600FF', ms=4)
    axd.set_title(r'(d) $\angle\varphi_z$'); axd.set_ylim(-185, 185)
    axd.set_xlabel('Position along gap [mm]')

    # (e) best theoretical overlay vs measured dominant shape (sign-restored)
    n = mode['mode_order']
    phi_dom = phi_y if mode['dom_component'] == 'y' else phi_z
    # de-rotate to a real shape using the magnitude-weighted global phase
    gphase = np.angle(np.sum(phi_dom * np.abs(phi_dom)) + 1e-30)
    meas_signed = np.real(phi_dom * np.exp(-1j * gphase))
    meas_signed = meas_signed / (np.max(np.abs(meas_signed)) + 1e-30)
    # flip theoretical sign to best match the measured shape for visual overlay
    th_sensors = theoretical_modes(md['xi'], md['params']['n_theory_modes'])[n - 1]
    sign = 1.0 if np.dot(meas_signed, th_sensors) >= 0 else -1.0
    axe.plot(md['xi'], meas_signed, 'o-', color=col, ms=4,
             label=f"measured ({mode['dom_component']})")
    axe.plot(md['theory_xi'], md['theory'][n - 1] * sign,
             '-', color='k', lw=1.2, alpha=0.7, label=f'theory mode {n}')
    axe.axhline(0, color='gray', lw=0.5, ls=':')
    axe.set_title(f'(e) best theory match (mode {n})')
    axe.set_xlabel('x / L'); axe.set_ylabel('norm. shape')
    axe.legend(fontsize=8)

    # (f) text summary
    axf.axis('off')
    macs = '  '.join(f'M{i+1}:{v:.2f}' for i, v in enumerate(mode['mac_vec']))
    txt = (f"f_n = {mode['f_n']:.1f} Hz\n"
           f"best theoretical mode: {n}\n"
           f"MAC = {mode['mac']:.3f}\n"
           f"MAC vector:\n  {macs}\n\n"
           f"polarization: {mode['dom_pol']}\n"
           f"axis ratio = {mode['dom_axis_ratio']:.3f}\n"
           f"handedness = {mode['dom_handedness']:+d}")
    axf.text(0.02, 0.98, txt, va='top', ha='left', family='monospace',
             fontsize=10, transform=axf.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    axf.set_title('(f) summary')

    fig.suptitle(f"{cfg['label']} — resonance {mode['f_n']:.1f} Hz",
                 fontsize=12)
    plt.tight_layout()
    return fig


def plot_mac_heatmap(cfg, ax=None):
    """Heatmap of MAC (measured resonances × theoretical modes) for one dataset."""
    md = _require_modal(cfg)
    if not md['modes']:
        return None
    M = np.array([m['mac_vec'] for m in md['modes']])
    n_theory = M.shape[1]
    if ax is None:
        _fig, ax = plt.subplots(figsize=(0.8 * n_theory + 2,
                                         0.5 * len(md['modes']) + 2))
    im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n_theory))
    ax.set_xticklabels([f'M{i+1}' for i in range(n_theory)])
    ax.set_yticks(range(len(md['modes'])))
    ax.set_yticklabels([f"{m['f_n']:.0f} Hz" for m in md['modes']])
    ax.set_xlabel('theoretical mode')
    ax.set_ylabel('measured resonance')
    ax.set_title(f"{cfg['label']} — MAC")
    for i in range(M.shape[0]):
        for j in range(n_theory):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if M[i, j] < 0.6 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8, label='MAC')
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# §6  Polarization-ellipse map
# ─────────────────────────────────────────────────────────────────────────────
def plot_ellipse_map(cfg, modes_per_row=None):
    """Polarization ellipses at every gap sensor, one subplot per resonance.

    Top-down hodogram view: x-axis is position along the cable; at each sensor
    the (y, z) particle-motion ellipse is drawn (y deflects horizontally, z
    vertically), scaled to the sensor spacing. Ellipse colour encodes the local
    polarization class.
    """
    md = _require_modal(cfg)
    modes = md['modes']
    if not modes:
        print(f"  [{cfg['label']}] no resonances to plot.")
        return None
    s_mm = md['s_mm']
    dx = np.median(np.diff(s_mm)) if len(s_mm) > 1 else 1.0
    p = md['params']

    n = len(modes)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n + 0.5), squeeze=False)
    axes = axes[:, 0]
    for ax, mode in zip(axes, modes):
        phi_y, phi_z = mode['phi_y'], mode['phi_z']
        major = mode['pol']['major']
        scale = 0.42 * dx / (major.max() + 1e-30)
        ratios = mode['pol']['axis_ratio']
        labels = classify_polarization(ratios, p['lin_thresh'], p['circ_thresh'])
        for i in range(len(s_mm)):
            yt, zt = ellipse_trace(phi_y[i], phi_z[i])
            ax.plot(s_mm[i] + yt * scale, zt * scale,
                    color=POL_COLORS[str(labels[i])], lw=1.0)
            ax.plot(s_mm[i], 0, '.', color='gray', ms=2)
        ax.set_ylabel(f"{mode['f_n']:.0f} Hz\n(z motion)")
        ax.set_title(f"mode {mode['mode_order']} — {mode['dom_pol']} "
                     f"(axis ratio {mode['dom_axis_ratio']:.2f}, "
                     f"hand {mode['dom_handedness']:+d})", fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')
    axes[-1].set_xlabel('Position along cable [mm]  (y motion = horizontal wiggle)')
    # legend
    handles = [plt.Line2D([], [], color=POL_COLORS[k], lw=2, label=k)
               for k in POL_MARKERS]
    axes[0].legend(handles=handles, fontsize=8, loc='upper right', ncol=3)
    fig.suptitle(f"{cfg['label']} — polarization ellipse map", fontsize=12)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# §6  3-D phase-snapshot montage (call explicitly for a chosen resonance)
# ─────────────────────────────────────────────────────────────────────────────
def plot_phase_snapshots(cfg, mode_index, scale=None, phases=(0.0, 0.25, 0.5, 0.75)):
    """3-D cable shape at four phases of one oscillation cycle for one resonance.

    Animated position = static XYZ + Re(φ · e^{i·2π·phase}) (transverse y, z),
    drawn over the gap sensors. `scale` (defaults to ~15 % of the chord length
    over the peak modal amplitude) exaggerates the small displacements.
    """
    md = _require_modal(cfg)
    mode = md['modes'][mode_index]
    gap_idx = md['gap_idx']
    XYZ = cfg['XYZ'][gap_idx]
    phi_y, phi_z = mode['phi_y'], mode['phi_z']

    amp = max(np.abs(phi_y).max(), np.abs(phi_z).max()) + 1e-30
    if scale is None:
        scale = 0.15 * cfg['chord_len'] / amp

    fig = plt.figure(figsize=(12, 9))
    for k, ph in enumerate(phases):
        e = np.exp(1j * 2.0 * np.pi * ph)
        dy = np.real(phi_y * e) * scale
        dz = np.real(phi_z * e) * scale
        x = XYZ[:, 0] * 1e3
        y = (XYZ[:, 1] + dy) * 1e3
        z = (XYZ[:, 2] + dz) * 1e3
        ax = fig.add_subplot(2, 2, k + 1, projection='3d')
        ax.plot(XYZ[:, 0] * 1e3, XYZ[:, 1] * 1e3, XYZ[:, 2] * 1e3,
                color='lightgray', lw=1.0, ls='--', label='static')
        ax.plot(x, y, z, 'o-', color=dataset_style(cfg)[0], ms=3, lw=1.4)
        ax.set_title(f'phase {ph*360:.0f}°')
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')
    fig.suptitle(f"{cfg['label']} — {mode['f_n']:.1f} Hz "
                 f"(mode {mode['mode_order']}, {mode['dom_pol']}) — "
                 f"phase snapshots ×{scale:.0f}", fontsize=12)
    plt.tight_layout()
    return fig


def deep_dive(cfg, with_heatmap=True, with_ellipses=True):
    """Run the full per-dataset visual deep dive (fingerprint + every mode)."""
    _require_modal(cfg)
    plot_fingerprint(cfg)
    plt.show()
    if with_heatmap and cfg['modal']['modes']:
        plot_mac_heatmap(cfg)
        plt.show()
    for i in range(len(cfg['modal']['modes'])):
        plot_mode_panels(cfg, i)
        plt.show()
    if with_ellipses and cfg['modal']['modes']:
        plot_ellipse_map(cfg)
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# §7  Cross-dataset summaries
# ─────────────────────────────────────────────────────────────────────────────
def build_summary_table(datasets):
    """Tidy DataFrame: one row per detected mode across all datasets.

    Columns: dataset, cable, gap_m, f_n, mode_order, MAC, polarization_type,
    axis_ratio, handedness.
    """
    import pandas as pd
    rows = []
    for cfg in datasets:
        md = cfg.get('modal')
        if md is None:
            continue
        for m in md['modes']:
            rows.append(dict(
                dataset=cfg['label'], cable=cfg.get('cable', ''),
                gap_m=cfg.get('gap_m', np.nan),
                f_n=round(m['f_n'], 2), mode_order=m['mode_order'],
                MAC=round(m['mac'], 3), polarization_type=m['dom_pol'],
                axis_ratio=round(m['dom_axis_ratio'], 3),
                handedness=m['dom_handedness'],
            ))
    return pd.DataFrame(rows)


def plot_mode_frequency_chart(datasets, ax=None):
    """Scatter: x = dataset, y = f_n (log), colour = mode order, marker = polarization."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(datasets) + 2), 6))
    labels = [cfg['label'] for cfg in datasets if cfg.get('modal')]
    x_of = {lbl: i for i, lbl in enumerate(labels)}

    cmap = plt.get_cmap('tab10')
    max_order = 1
    for cfg in datasets:
        md = cfg.get('modal')
        if md is None:
            continue
        xi = x_of[cfg['label']]
        for m in md['modes']:
            max_order = max(max_order, m['mode_order'])
            ax.scatter(xi, m['f_n'], s=90,
                       color=cmap((m['mode_order'] - 1) % 10),
                       marker=POL_MARKERS.get(m['dom_pol'], 'o'),
                       edgecolor='k', linewidth=0.4, zorder=3)

    ax.set_yscale('log')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel('Resonance frequency [Hz]')
    ax.set_title('Detected resonances across datasets')
    ax.grid(True, which='both', alpha=0.25)

    order_handles = [plt.Line2D([], [], marker='o', ls='', color=cmap((k) % 10),
                                markeredgecolor='k', label=f'mode {k+1}')
                     for k in range(max_order)]
    pol_handles = [plt.Line2D([], [], marker=POL_MARKERS[k], ls='', color='gray',
                              markeredgecolor='k', label=k) for k in POL_MARKERS]
    leg1 = ax.legend(handles=order_handles, title='mode order',
                     fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=pol_handles, title='polarization',
              fontsize=8, loc='lower left', bbox_to_anchor=(1.01, 0.0))
    plt.tight_layout()
    return ax


def plot_fn_vs_gap(datasets, ax=None, max_order=None):
    """f_n vs gap length (log-log) with theoretical L⁻² reference lines per mode.

    Reference lines are anchored to the geometric-mean (gap, f_n) of all points
    of each mode order, then scaled as f ∝ L⁻², giving the expected slope for a
    fixed-tension / fixed-bending-stiffness beam.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 6))

    # collect points per mode order
    pts = {}
    for cfg in datasets:
        md = cfg.get('modal')
        if md is None:
            continue
        col, marker, _ = dataset_style(cfg)
        gap_cm = cfg['gap_m'] * 100.0
        for m in md['modes']:
            pts.setdefault(m['mode_order'], []).append((gap_cm, m['f_n']))
            ax.scatter(gap_cm, m['f_n'], s=70, color=col, marker=marker,
                       edgecolor='k', linewidth=0.4, zorder=3)
            ax.annotate(cfg['label'].replace('_', ' '),
                        (gap_cm, m['f_n']), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')

    if max_order is None:
        max_order = max(pts) if pts else 1
    gmin = min((g for v in pts.values() for g, _ in v), default=5)
    gmax = max((g for v in pts.values() for g, _ in v), default=15)
    g_line = np.linspace(gmin * 0.9, gmax * 1.1, 50)
    cmap = plt.get_cmap('tab10')
    for order, vals in sorted(pts.items()):
        if order > max_order:
            continue
        g = np.array([v[0] for v in vals])
        f = np.array([v[1] for v in vals])
        # anchor: geometric mean → C = f̄ · ḡ²  so f = C / L²
        C = np.exp(np.mean(np.log(f) + 2.0 * np.log(g)))
        ax.plot(g_line, C / g_line ** 2, ls='--', lw=1.0,
                color=cmap((order - 1) % 10), alpha=0.8,
                label=f'mode {order}  ∝ L⁻²')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Gap length [cm]')
    ax.set_ylabel('Resonance frequency [Hz]')
    ax.set_title('f_n vs gap length with L⁻² scaling')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout()
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# §8  Interactive dashboard
# ─────────────────────────────────────────────────────────────────────────────
def modal_dashboard(datasets):
    """Pick a dataset → pick a detected resonance → show its deep-dive plots.

    Requires modal.analyze_dataset to have been run on each dataset (results in
    cfg['modal']); datasets without results are analysed on demand with defaults.
    """
    import ipywidgets as ipw
    from IPython.display import display

    sty = {"description_width": "initial"}
    label_to_cfg = {cfg['label']: cfg for cfg in datasets}

    cable_w = ipw.Dropdown(options=list(label_to_cfg), description='Dataset:', style=sty)
    res_w = ipw.Dropdown(description='Resonance:', style=sty)
    view_w = ipw.ToggleButtons(
        options=['fingerprint', 'mode panels', 'MAC heatmap',
                 'ellipse map', 'phase snapshots'],
        value='mode panels', description='View:', style=sty)
    out = ipw.Output()

    def _modes(cfg):
        if 'modal' not in cfg:
            analyze_dataset(cfg, verbose=False)
        return cfg['modal']['modes']

    def _sync_res(*_):
        cfg = label_to_cfg[cable_w.value]
        modes = _modes(cfg)
        res_w.options = [(f"{m['f_n']:.1f} Hz  (M{m['mode_order']}, {m['dom_pol']})", i)
                         for i, m in enumerate(modes)] or [('—', None)]
        res_w.value = res_w.options[0][1]

    def update(_=None):
        cfg = label_to_cfg[cable_w.value]
        modes = _modes(cfg)
        with out:
            out.clear_output(wait=True)
            view = view_w.value
            if view == 'fingerprint':
                plot_fingerprint(cfg); plt.show()
            elif view == 'MAC heatmap':
                if modes:
                    plot_mac_heatmap(cfg); plt.show()
                else:
                    print('No resonances detected.')
            elif view == 'ellipse map':
                plot_ellipse_map(cfg); plt.show()
            elif res_w.value is None:
                print('No resonances detected for this dataset.')
            elif view == 'mode panels':
                plot_mode_panels(cfg, res_w.value); plt.show()
            elif view == 'phase snapshots':
                plot_phase_snapshots(cfg, res_w.value); plt.show()

    cable_w.observe(lambda c: (_sync_res(), update()), names='value')
    res_w.observe(update, names='value')
    view_w.observe(update, names='value')

    _sync_res()
    display(ipw.VBox([ipw.HBox([cable_w, res_w]), view_w, out]))
    update()

# ─────────────────────────────────────────────────────────────────────────────
# Shaker input-bias diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def plot_shaker_directionality(cfg, f_min=None, f_max=None, smooth_hz=2.0,
                               figsize=(11, 8)):
    """Quantify how much the shaker head excites off-axis (y/z) per frequency.

    Three stacked panels sharing the frequency axis:
        (1) shaker-head amplitude spectra |V_x|, |V_y|, |V_z| (log y) — the
            actual input the cable receives;
        (2) power fractions  P_i / (P_x+P_y+P_z)  — the directionality; the
            drive is clean where the x-fraction ≈ 1 and biased where the
            y/z fractions rise;
        (3) the cable's total-power detection fingerprint with the detected
            resonances marked — laid over the same axis so peaks coinciding
            with off-axis input bands are immediately visible.

    Requires cfg['modal'] (run modal.analyze_dataset first). smooth_hz applies
    a boxcar to the shaker spectra for readability.
    """
    from .modal import shaker_input_spectrum, smooth_spectrum
    md = _require_modal(cfg)
    p = md['params']
    if f_min is None:
        f_min = p['f_min']
    if f_max is None:
        f_max = p['f_max']

    sh = shaker_input_spectrum(cfg, method='welch')
    f = sh['freqs']
    band = (f >= f_min) & (f <= f_max)

    fig, (ax_a, ax_f, ax_c) = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [1.6, 1.2, 1.2]})

    cols = {'x': '#404040', 'y': '#0022FF', 'z': '#E600FF'}
    for comp in ('x', 'y', 'z'):
        amp = smooth_spectrum(f, sh[f'amp_{comp}'], smooth_hz)
        ax_a.semilogy(f[band], amp[band] * 1e3, lw=1.0, color=cols[comp],
                      label=f'$|V_{comp}|$ shaker head')
    ax_a.set_ylabel('shaker velocity ASD [mm/s/√Hz]')
    ax_a.legend(fontsize=8, loc='upper right')

    for comp in ('x', 'y', 'z'):
        frac = smooth_spectrum(f, sh[f'frac_{comp}'], smooth_hz)
        ax_f.plot(f[band], frac[band], lw=1.2, color=cols[comp],
                  label=f'{comp} fraction')
    ax_f.axhline(0.5, color='k', lw=0.6, ls=':', alpha=0.5)
    ax_f.set_ylabel('input power fraction')
    ax_f.set_ylim(0, 1.02)
    ax_f.legend(fontsize=8, loc='center right')

    fb = md['freqs']
    bandc = (fb >= f_min) & (fb <= f_max)
    ax_c.plot(fb[bandc], md['sig_norm'][bandc], color='gray', lw=1.0,
              label='cable detection signal (norm.)')
    for fpk in md['peaks']:
        ax_c.axvline(fpk, color='r', ls=':', lw=1.0, alpha=0.8)
    ax_c.set_ylabel('cable fingerprint')
    ax_c.set_xlabel('frequency [Hz]')
    ax_c.legend(fontsize=8, loc='upper right')

    ax_a.set_xlim(f_min, f_max)
    ax_a.set_title(f"{cfg.get('label', '?')} — shaker directionality vs. "
                   "detected cable resonances")
    plt.tight_layout()
    return fig
