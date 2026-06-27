"""Plot helpers: per-frequency QC dashboard, raw traces, spectra, spectrograms."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .config import dataset_style, theta_from_sag, theta_from_material
from .geometry import compute_chord, measure_sag, sag_method_label, project_onto_chord
from .signal import amplitude_spectrum, compute_spectrogram, compute_fk_spectrum


# ─────────────────────────────────────────────────────────────────────────────
# Existing helpers (used by per-frequency QC + summary)
# ─────────────────────────────────────────────────────────────────────────────
def plot_spacetime(ax, x, t, data, vmax=None, cmap='RdBu_r', unit='[m/s]'):
    """Space-time pcolormesh with descending time axis (seismic-shotgather style)."""
    if vmax is None:
        vmax = np.nanmax(np.abs(data)) * 0.95
    if vmax == 0:
        vmax = 1e-30
    pcm = ax.pcolormesh(x * 1e3, t, data, cmap=cmap, vmin=-vmax, vmax=vmax,
                        shading='nearest')
    ax.invert_yaxis()
    ax.set_xlabel('Position [mm]')
    ax.set_ylabel('Time [s]')
    plt.colorbar(pcm, ax=ax, label=unit, shrink=0.85)


def plot_initial_geometry(datasets, sag_use_3d=True, sag_use_parabola=False):
    """Plot 3-D shape, side/top view + summary box for each dataset.

    Also annotates cfg with chord_unit, chord_len, chord_start, static_sag,
    idx_sag, theta_pred, eta_pred.

    Parameters
    ----------
    sag_use_3d      : True = full 3-D perpendicular distance from chord
                      (default).  False = x-z plane projection only.
    sag_use_parabola: True = fit a parabola through the cable points and use
                      the parabola peak as sag, mitigating outliers.
    """
    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 3, figsize=(15, 3.5 * n_ds), squeeze=False)
    method_lbl = sag_method_label(sag_use_3d, sag_use_parabola)

    for row, cfg in enumerate(datasets):
        XYZ = cfg['XYZ']
        IL, IR = cfg['idx_left'], cfg['idx_right']
        col, _, _fs = dataset_style(cfg)

        e_hat, L_chord, P1 = compute_chord(XYZ, IL, IR)
        sag, idx_sag, perp = measure_sag(XYZ, IL, IR,
                                          use_3d=sag_use_3d,
                                          use_parabola=sag_use_parabola)

        cable_name = cfg.get('cable')
        theta_sag = theta_from_sag(cable_name, sag)
        if theta_sag is None:
            theta_sag = 0.5 * (sag / cfg['radius_m']) ** 2
        theta_mat, theta_mat_std = theta_from_material(cable_name, L_chord)
        cfg.update(dict(
            chord_unit=e_hat, chord_len=L_chord, chord_start=P1,
            static_sag=sag, idx_sag=idx_sag,
            theta_pred=theta_sag,
            eta_pred=1.0 / (1.0 + theta_sag),
            theta_material=theta_mat,
            theta_material_std=theta_mat_std,
            eta_material=(None if theta_mat is None else 1.0 / (1.0 + theta_mat)),
        ))

        # Side view (x-z plane)
        ax = axes[row, 0]
        ax.plot(XYZ[:, 0] * 1e3, XYZ[:, 2] * 1e3, 'o-', color=col, ms=4)
        ax.plot([XYZ[IL, 0] * 1e3, XYZ[IR, 0] * 1e3],
                [XYZ[IL, 2] * 1e3, XYZ[IR, 2] * 1e3], 'r--', lw=0.8, label='Chord')

        # If parabola mode: overlay the constrained parabola in x-z and mark vertex
        if sag_use_parabola:
            pts_xz = XYZ[IL:IR + 1]
            x_pts = pts_xz[:, 0]
            z_pts = pts_xz[:, 2]
            if len(x_pts) >= 3:
                x_L, z_L = x_pts[0], z_pts[0]
                x_R, z_R = x_pts[-1], z_pts[-1]
                dx_chord = x_R - x_L
                # Constrained fit: Δz = a_c * (x-x_L) * (x-x_R)
                # passes exactly through both endpoints
                if abs(dx_chord) > 1e-12:
                    slope = (z_R - z_L) / dx_chord
                    z_chord_pts = z_L + slope * (x_pts - x_L)
                    delta_z = z_pts - z_chord_pts
                    phi_xz = (x_pts - x_L) * (x_pts - x_R)
                    denom_xz = float(np.dot(phi_xz, phi_xz))
                    a_c = (float(np.dot(phi_xz, delta_z) / denom_xz)
                           if denom_xz > 0 else 0.0)
                    x_fit = np.linspace(x_L, x_R, 200)
                    z_fit = (z_L + slope * (x_fit - x_L)
                             + a_c * (x_fit - x_L) * (x_fit - x_R))
                    ax.plot(x_fit * 1e3, z_fit * 1e3, 'm-', lw=1.2, alpha=0.7,
                            label='Parabola fit')
                    # Point of maximum perpendicular distance from the chord.
                    # For the constrained parabola this is always at x = midpoint,
                    # regardless of chord tilt (d(x) ∝ (x-xL)(xR-x), max at mid).
                    if a_c != 0:
                        x_mid = (x_L + x_R) / 2.0
                        z_mid = float(z_L + slope * (x_mid - x_L)
                                      + a_c * (x_mid - x_L) * (x_mid - x_R))
                        ax.plot(x_mid * 1e3, z_mid * 1e3, '*',
                                color='magenta', ms=14,
                                markeredgecolor='black', markeredgewidth=0.5,
                                zorder=6, label=f'w0 = {sag*1e3:.2f} mm')

        sag_lbl = 'Nearest sensor' if sag_use_parabola else f'w0 = {sag*1e3:.2f} mm'
        ax.plot(XYZ[idx_sag, 0] * 1e3, XYZ[idx_sag, 2] * 1e3, 's',
                color='red', ms=7, label=sag_lbl)
        ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
        ax.set_title(f"{cfg['label']} - side view")
        ax.legend(fontsize=8)

        # Top view (x-y plane) — also mark sag point
        ax = axes[row, 1]
        ax.plot(XYZ[:, 0] * 1e3, XYZ[:, 1] * 1e3, 'o-', color=col, ms=4)
        ax.plot(XYZ[idx_sag, 0] * 1e3, XYZ[idx_sag, 1] * 1e3, 's',
                color='red', ms=7)
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
        ax.set_title(f"{cfg['label']} - top view")

        # Summary box
        ax = axes[row, 2]
        ax.axis('off')
        from .config import CABLE_PROPERTIES
        cable_name = cfg.get('cable', '')
        cprops = CABLE_PROPERTIES.get(cable_name, {})
        if cprops.get('cross_section') == 'rectangular':
            w_mm = cprops['width_m'] * 1e3
            h_mm = cprops['height_m'] * 1e3
            geom_line = f"Section      : rect {w_mm:.2f}×{h_mm:.2f} mm ({cprops.get('bending_axis','weak')} axis)"
        else:
            geom_line = f"Radius r     : {cfg['radius_m']*1e3:.2f} mm"

        if cfg.get('theta_material') is not None:
            _ts, _ss = cfg['theta_material'], cfg.get('theta_material_std') or 0.0
            theta_mat_str = f"{_ts:.4f} ± {_ss:.4f}"
        else:
            theta_mat_str = "— (ρ, E not set)"
        eta_mat_str = (f"{cfg['eta_material']:.3f}" if cfg.get('eta_material') is not None
                       else "—")
        txt = (
            f"Cable        : {cfg['label']}\n"
            f"Nominal gap  : {cfg['gap_m']*100:.1f} cm\n"
            f"Chord length : {L_chord*100:.2f} cm\n"
            f"Chord e_hat  : ({e_hat[0]:+.3f}, {e_hat[1]:+.3f}, {e_hat[2]:+.3f})\n"
            f"{geom_line}\n\n"
            f"Sag method   : {method_lbl}\n"
            f"Static sag w0: {sag*1e3:.3f} mm\n\n"
            f"— Sag-based —\n"
            f"Theta = A·w0²/(8I) = {cfg['theta_pred']:.4f}\n"
            f"eta_pred       = {cfg['eta_pred']:.3f}\n\n"
            f"— Material-based —\n"
            f"Theta_mat      = {theta_mat_str}\n"
            f"eta_mat        = {eta_mat_str}"
        )
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
                va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
        ax.set_title(f"{cfg['label']} - geometry summary")

    plt.suptitle(f'Initial geometry & static sag  [{method_lbl}]',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_per_frequency_qc(cfg, result, ref='shaker'):
    """Multi-panel figure for one (cable, frequency) pair.

    ref : 'shaker' or 'ends'
    """
    f_target = result['f_target']
    t_win = result['t_win']
    x = cfg['x']
    col, _, _fs = dataset_style(cfg)

    xc = 0.5 * (x[cfg['idx_left']:cfg['idx_right']]
                + x[cfg['idx_left'] + 1:cfg['idx_right'] + 1])

    if ref == 'ends':
        delta_ref = result['delta_ends']
        delta_ref_lbl = 'Δu_ends (sensors)'
        eta_t = result['eta_t_ends']
        eta_med = result['eta_med_ends']
        eps_ref_win = result['eps_ref_ends']
        ref_label = f'ε_ref = Δu_ends / L_ends ({result["L_ends"]*1e3:.1f} mm)'
        eta_t_m3 = result['eta_t_m3_ends']
        eta_med_m3 = result['eta_med_m3_ends']
    elif ref == 'shaker_end':
        delta_ref = result['delta_shend']
        delta_ref_lbl = 'δL_shend (shaker−end)'
        eta_t = result['eta_t_shend']
        eta_med = result['eta_med_shend']
        eps_ref_win = result['eps_ref_shend']
        ref_label = f'ε_ref = δL_shend / L_shend ({result["L_shend"]*1e3:.1f} mm)'
        eta_t_m3 = result['eta_t_m3_shend']
        eta_med_m3 = result['eta_med_m3_shend']
    else:
        delta_ref = result['delta_L_win']
        delta_ref_lbl = 'δL (shaker)'
        eta_t = result['eta_t']
        eta_med = result['eta_med']
        eps_ref_win = result['delta_L_win'] / cfg['chord_len']
        ref_label = 'ε_ref = δL / L_chord (shaker)'
        eta_t_m3 = result['eta_t_m3']
        eta_med_m3 = result['eta_med_m3']

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.35)

    # Row 1 — velocity wavefields
    ax = fig.add_subplot(gs[0, 0]); plot_spacetime(ax, x, t_win, result['vx'], unit='[m/s]'); ax.set_title('$v_x$')
    ax = fig.add_subplot(gs[0, 1]); plot_spacetime(ax, x, t_win, result['vy'], unit='[m/s]'); ax.set_title('$v_y$')
    ax = fig.add_subplot(gs[0, 2]); plot_spacetime(ax, x, t_win, result['vz'], unit='[m/s]'); ax.set_title('$v_z$')

    # Row 1 col 4 — spectrum at midpoint
    mid = (cfg['idx_left'] + cfg['idx_right']) // 2
    ax = fig.add_subplot(gs[0, 3])
    for sig, lbl, c in [
        (result['vx'][:, mid], '$v_x$', 'C0'),
        (result['vy'][:, mid], '$v_y$', 'C1'),
        (result['vz'][:, mid], '$v_z$', 'C2'),
    ]:
        freqs, amp = amplitude_spectrum(sig[:, np.newaxis], cfg['fs'])
        ax.plot(freqs, amp.ravel() * 1e3, color=c, lw=1.0, label=lbl)
    ax.set_xlim(result['f_lo'] * 0.5, result['f_hi'] * 1.5)
    ax.axvline(f_target, color='k', ls=':', lw=0.8)
    ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('|V| [mm/s]')
    ax.set_title(f'Spectrum at sensor {mid}')
    ax.legend(fontsize=8)

    # Row 2 — displacement wavefields
    u_axial = project_onto_chord(result['ux'], result['uy'], result['uz'], cfg['chord_unit'])
    ax = fig.add_subplot(gs[1, 0]); plot_spacetime(ax, x, t_win, u_axial, unit='[m]'); ax.set_title('$u_{\\parallel}$ (chord-projected)')
    ax = fig.add_subplot(gs[1, 1]); plot_spacetime(ax, x, t_win, result['uy'], unit='[m]'); ax.set_title('$u_y$')
    ax = fig.add_subplot(gs[1, 2]); plot_spacetime(ax, x, t_win, result['uz'], unit='[m]'); ax.set_title('$u_z$')

    # Row 2 col 4 — cable elongation vs reference
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(t_win, delta_ref * 1e6, color='k', lw=1.0, label=delta_ref_lbl)
    ax.plot(t_win, result['delta_xl'] * 1e6, color=col, lw=1.0, label='δxₗ (arc-length)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Displacement [μm]')
    ax.set_title('Cable elongation vs reference')
    ax.legend(fontsize=8)

    # Row 3 — strain wavefields
    ax = fig.add_subplot(gs[2, 0]); plot_spacetime(ax, x, t_win, result['strain_grad'], unit='[m/m]')
    ax.set_title('Method 1: spatial gradient  ε = ∂u/∂s')
    ax = fig.add_subplot(gs[2, 1]); plot_spacetime(ax, xc, t_win, result['seg_strain'], unit='[m/m]')
    ax.set_title('Method 2: per-segment δd/L₀ (3-D arc-length)')

    # Row 3 col 3 — reference vs measured mean strain
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t_win, eps_ref_win * 1e6, 'k', lw=1.0, label=ref_label)
    eps_meas_mean = result['seg_strain'].mean(axis=1)
    ax.plot(t_win, eps_meas_mean * 1e6, color=col, lw=1.0, label='⟨ε⟩ cable (arc-length)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Strain [μm/m]')
    ax.set_title('Reference vs measured (gap-averaged)')
    ax.legend(fontsize=8)

    # Row 3 col 4 — instantaneous eta
    ax = fig.add_subplot(gs[2, 3])
    ax.plot(t_win, eta_t, color=col, lw=1.0)
    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax.axhline(cfg['eta_pred'], color='red', ls=':', lw=1.0, label=f"η_pred = {cfg['eta_pred']:.3f}")
    ax.axhline(eta_med, color=col, ls='-.', lw=1.0, label=f"η_med = {eta_med:.3f}")
    ax.set_ylim(-0.5, 2.0)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('η = δxₗ / δL')
    ax.set_title('Instantaneous coupling efficiency')
    ax.legend(fontsize=8)

    # Row 4 — Method 3 (Fourier-domain spatial gradient)
    ax = fig.add_subplot(gs[3, 0])
    plot_spacetime(ax, x, t_win, result['strain_fourier'], unit='[m/m]')
    ax.set_title('Method 3: Fourier gradient  ε = ∂u/∂s (k-domain)')

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t_win, delta_ref * 1e6, 'k', lw=1.0, ls='--', label=delta_ref_lbl)
    ax.plot(t_win, result['delta_xl'] * 1e6, color='C0', lw=1.0, label='M2 arc-len')
    ax.plot(t_win, result['delta_xl_m1'] * 1e6, color='C1', lw=1.0, label='M1 gradient')
    ax.plot(t_win, result['delta_xl_m3'] * 1e6, color='C2', lw=1.0, label='M3 Fourier')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Elongation [μm]')
    ax.set_title('All 3 methods vs reference')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 2])
    ax.plot(t_win, eps_ref_win * 1e6, 'k', lw=1.0, label=ref_label)
    eps_m3_mean = result['strain_fourier'][
        :, cfg['idx_left']:cfg['idx_right'] + 1].mean(axis=1)
    ax.plot(t_win, eps_m3_mean * 1e6, color=col, lw=1.0, label='⟨ε⟩ cable (Fourier)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Strain [μm/m]')
    ax.set_title('Reference vs M3 (gap-averaged)')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 3])
    ax.plot(t_win, eta_t_m3, color=col, lw=1.0)
    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax.axhline(cfg['eta_pred'], color='red', ls=':', lw=1.0,
               label=f"η_pred = {cfg['eta_pred']:.3f}")
    ax.axhline(eta_med_m3, color=col, ls='-.', lw=1.0,
               label=f"η_med = {eta_med_m3:.3f}")
    ax.set_ylim(-0.5, 2.0)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('η = δxₗ / δL')
    ax.set_title('η(t) — Method 3 (Fourier grad.)')
    ax.legend(fontsize=8)

    plt.suptitle(
        f"{cfg['label']}  -  f = {f_target} Hz  "
        f"(bandpass {result['f_lo']:.1f}-{result['f_hi']:.1f} Hz)  "
        f"[ref: {ref}]",
        fontsize=13, y=1.00)
    plt.show()

def plot_coupling_summary(datasets, ref='shaker', logx=True, ylim=(-0.2, 2.0)):
   # """One η_med(f) line per dataset; dotted horizontal lines = quasi-static prediction."""
    if ref == 'ends':
        eta_key = 'eta_med_ends'
        ref_label = 'Δu_ends'
    elif ref == 'shaker_end':
        eta_key = 'eta_med_shend'
        ref_label = 'δL_shend'
    else:
        eta_key = 'eta_med'
        ref_label = 'δL (shaker)'

    fig, ax = plt.subplots(figsize=(10, 5))
    for cfg in datasets:
        col, marker, fillstyle = dataset_style(cfg)
        fs_arr, eta_arr = [], []
        for f, res in cfg['per_freq'].items():
            fs_arr.append(f); eta_arr.append(res[eta_key])
        ax.plot(fs_arr, eta_arr, marker=marker, linestyle='-', color=col,
                ms=7, fillstyle=fillstyle, label=cfg['label'])
    #    ax.axhline(cfg['eta_pred'], color=col, ls=':', lw=0.8, alpha=0.7)

    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax.set_xscale('log' if logx else 'linear')
    ax.set_xlabel('Target frequency [Hz]')
    ax.set_ylabel(f'Median η = δxₗ / {ref_label} within window')
    ax.set_title(f'Coupling efficiency vs frequency  [ref: {ref}]\n'
                 'Colour = cable  |  ◆ 5 cm  ◼ 10 cm  ● 15 cm  |  no-fill = Sag')
    ax.set_ylim(ylim)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    for cfg in datasets:
        print(f"{cfg['label']}   (Theta={cfg['theta_pred']:.3f}, eta_pred={cfg['eta_pred']:.3f})")
       # for f, res in cfg['per_freq'].items():
       #     print(f"   f = {f:5.0f} Hz   eta_med = {res[eta_key]:+.3f}")
        print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# New helpers — spectral analysis notebook
# ─────────────────────────────────────────────────────────────────────────────
_COMPONENT_LABELS = {
    'vx': '$v_x$', 'vy': '$v_y$', 'vz': '$v_z$',
    'ux': '$u_x$', 'uy': '$u_y$', 'uz': '$u_z$',
}

_COMPONENT_COLORS = {
    'vx': 'red', 'vy': 'green', 'vz': 'blue',
    'ux': 'red', 'uy': 'green', 'uz': 'blue',
}


def _get_component_array(cfg_or_ref, name):
    """Return a (N_t, N_s) array for cable cfg or (N_t,)→(N_t,1) for shaker ref."""
    arr = cfg_or_ref[name]
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    return arr


def plot_raw_traces_uniform(cfg, components=('vx', 'vy', 'vz'),
                            sensor_indices=None, share_scale='dataset',
                            unit_scale=1e3, unit_label='[mm/s]',
                            title=None, max_rows=None):
    """Plot raw traces in a grid (rows = sensors, cols = components) on a
    uniform amplitude scale so the dominant component is visually obvious.

    Parameters
    ----------
    cfg : dataset cfg dict (loaded) or a shaker reference dict from
          io.load_shaker_reference (single 'sensor').
    components : tuple of component keys present in cfg ('vx','vy','vz', ...).
    sensor_indices : iterable of sensor indices; default = all sensors.
        For a shaker-reference dict, this is ignored.
    share_scale : 'dataset' → one y-limit across all subplots
                  'component' → one y-limit per column
                  'subplot'   → each subplot autoscales
    unit_scale, unit_label : multiplier and label for the y-axis.
    """
    t = cfg['t']
    arrs = [_get_component_array(cfg, c) for c in components]
    n_sensors_total = arrs[0].shape[1]
    if sensor_indices is None:
        sensor_indices = list(range(n_sensors_total))
    if max_rows is not None and len(sensor_indices) > max_rows:
        sensor_indices = sensor_indices[:max_rows]

    if share_scale == 'dataset':
        vmax = max(np.abs(a[:, sensor_indices]).max() for a in arrs)
        vmaxes = [vmax] * len(components)
    elif share_scale == 'component':
        vmaxes = [np.abs(a[:, sensor_indices]).max() for a in arrs]
    else:
        vmaxes = [None] * len(components)
    vmaxes = [v if (v is not None and v > 0) else None for v in vmaxes]

    n_rows = len(sensor_indices)
    n_cols = len(components)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 1.4 * n_rows + 0.5),
                             sharex=True, squeeze=False)

    for i, s_idx in enumerate(sensor_indices):
        for j, comp in enumerate(components):
            ax = axes[i, j]
            ax.plot(t, arrs[j][:, s_idx] * unit_scale, lw=0.5,
                    color=_COMPONENT_COLORS.get(comp, f'C{j}'))
            if vmaxes[j] is not None:
                ax.set_ylim(-vmaxes[j] * unit_scale * 1.05,
                            vmaxes[j] * unit_scale * 1.05)
            if i == 0:
                ax.set_title(_COMPONENT_LABELS.get(comp, comp), fontsize=11)
            if j == 0:
                ax.set_ylabel(f's{s_idx}\n{unit_label}', fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel('Time [s]')

    if title is None:
        title = cfg.get('label', cfg.get('path', 'raw traces'))
    fig.suptitle(f"Raw traces — {title}  ({share_scale} amplitude scale)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_amplitude_spectra(cfg, components=('vx', 'vy', 'vz'),
                           sensor_indices=None, f_max=None,
                           xscale='log', yscale='log',
                           unit_scale=1e3, unit_label='|V| [mm/s]',
                           overlay_sensors=True, title=None):
    """Amplitude spectra per component.

    overlay_sensors=True : one panel per component, all sensors overlaid (faint).
                          Average across sensors plotted in bold.
    overlay_sensors=False: grid (rows=sensors, cols=components).
    """
    fs = cfg['fs']
    arrs = [_get_component_array(cfg, c) for c in components]
    n_sensors_total = arrs[0].shape[1]
    if sensor_indices is None:
        sensor_indices = list(range(n_sensors_total))

    spectra = []
    for a in arrs:
        freqs, spec = amplitude_spectrum(a[:, sensor_indices], fs)
        spectra.append((freqs, spec))

    if overlay_sensors:
        n_cols = len(components)
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.5),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes[0]
        for j, comp in enumerate(components):
            ax = axes[j]
            freqs, spec = spectra[j]
            col = _COMPONENT_COLORS.get(comp, f'C{j}')
            for k, s_idx in enumerate(sensor_indices):
                ax.plot(freqs, spec[:, k] * unit_scale, color=col,
                        lw=0.5, alpha=0.35)
            mean_spec = spec.mean(axis=1)
            ax.plot(freqs, mean_spec * unit_scale, color=col,
                    lw=1.6, label=f'mean ({len(sensor_indices)} sens.)')
            ax.set_title(_COMPONENT_LABELS.get(comp, comp))
            ax.set_xlabel('Frequency [Hz]')
            if j == 0:
                ax.set_ylabel(unit_label)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            if f_max is not None:
                ax.set_xlim(right=f_max)
            ax.legend(fontsize=8, loc='best')
    else:
        n_rows = len(sensor_indices)
        n_cols = len(components)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.6 * n_cols, 1.6 * n_rows + 0.5),
                                 sharex=True, sharey=True, squeeze=False)
        for j, comp in enumerate(components):
            freqs, spec = spectra[j]
            col = _COMPONENT_COLORS.get(comp, f'C{j}')
            for i, s_idx in enumerate(sensor_indices):
                ax = axes[i, j]
                ax.plot(freqs, spec[:, i] * unit_scale, color=col, lw=0.7)
                ax.set_xscale(xscale); ax.set_yscale(yscale)
                if f_max is not None:
                    ax.set_xlim(right=f_max)
                if i == 0:
                    ax.set_title(_COMPONENT_LABELS.get(comp, comp))
                if j == 0:
                    ax.set_ylabel(f's{s_idx}', fontsize=9)
                if i == n_rows - 1:
                    ax.set_xlabel('Frequency [Hz]')

    if title is None:
        title = cfg.get('label', cfg.get('path', 'amplitude spectra'))
    fig.suptitle(f"Amplitude spectra — {title}", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


def _eta_theta_theory(ax, percent=True, show_theory=True):
    """Overlay the η = 1/(1+Θ) theory curve, Θ=1/32 marker and 100 % line."""
    if not show_theory:
        return
    th = np.logspace(-5, 3, 800)
    ax.plot(th, 100.0 / (1.0 + th) if percent else 1.0 / (1.0 + th),
            'k-', lw=2, alpha=0.8, zorder=1,
            label=r'$\eta = 1/(1+\Theta)$ (theory)')
    ax.axvline(1.0 / 32.0, color='gray', ls=':', lw=1.2, alpha=0.7,
               label=r'$\Theta = 1/32$  ($w_0 = r/4$)')
    ax.axhline(100.0 if percent else 1.0, color='gray', ls='--',
               lw=0.7, alpha=0.5)


def _eta_theta_legend(ax):
    """Build the shared cable/gap/sag legend (prepended with theory handles)."""
    from .config import CABLE_COLORS, GAP_MARKERS
    from matplotlib.lines import Line2D

    def _mh(marker, color, fillstyle, label, mew=0.7, mec='black', ms=10):
        kw = dict(marker=marker, linewidth=0, markersize=ms, color='none',
                  markerfacecolor=color, markeredgecolor=mec,
                  markeredgewidth=mew, fillstyle=fillstyle, label=label)
        if fillstyle == 'none':
            kw['markerfacecolor'] = 'none'
            kw['markeredgecolor'] = color
            kw['markeredgewidth'] = 1.5
        return Line2D([0], [0], **kw)

    sep = Line2D([0], [0], color='none', label=' ')
    cable_handles = [_mh('o', col, 'full', name, mec='black', mew=0.5)
                     for name, col in CABLE_COLORS.items()]
    gap_handles = [_mh(mkr, '#888888', 'full', f'{int(g*100)} cm', mec='black', mew=0.5)
                   for g, mkr in GAP_MARKERS.items()]
    sag_handles = [
        _mh('o', '#555555', 'full', 'No sag', mec='black', mew=0.5),
        _mh('o', '#555555', 'none', 'Sag (intended)', mec='#555555', mew=1.5),
    ]
    theory_handles = list(ax.get_legend_handles_labels()[0])
    all_handles = (theory_handles + [sep] + cable_handles + [sep]
                   + gap_handles + [sep] + sag_handles)
    ax.legend(handles=all_handles, fontsize=8, loc='upper right',
              framealpha=0.9, ncol=1)


def plot_eta_vs_theta(datasets, f_min=0.0, f_max=50.0, ref='shaker',
                       xlim=None, ylim=None, percent=True,
                       show_errorbars=True, show_theory=True,
                       bw_frac=None, grad_direction='chord',
                       strain_method='arclength',
                       estimator='instantaneous',
                       theta_source='sag',
                       verbose=False):
    """Scatter plot of mean coupling efficiency η vs Θ.

    Each dataset is one marker: colour = cable, shape = gap size (5/10/15 cm),
    fillstyle = 'full' (no intended sag) or 'none' (Sag variant).
    Overlaid with the theoretical curve eta = 1/(1+Theta).

    Parameters
    ----------
    datasets      : list of loaded cable cfg dicts (geometry must be computed).
    f_min, f_max  : frequency band [Hz] over which eta is averaged.
    ref           : 'shaker' or 'ends' — which reference elongation to use.
    xlim, ylim    : optional (lo, hi) axis limits.  ylim is in percent if
                    percent=True, else fractional.
    percent       : if True, y-axis in % (0–100); else fractional (0–1).
    show_errorbars: draw vertical bars showing std of eta across freq. band.
    show_theory   : overlay the theoretical curve and the Theta=1/32 marker.
    bw_frac       : bandpass half-width; None uses config.BANDWIDTH_FRAC.
    grad_direction: 'chord' or 'cartesian'.
    strain_method : 'arclength' (default) or 'gradient' — which elongation estimator.
    estimator     : 'instantaneous' (default) or 'envelope' — how η is computed.
                    'envelope' uses Hilbert envelopes so the ratio is phase-insensitive.
    theta_source  : 'sag'      — use Θ from measured sag, A·w₀²/(8I)  [default]
                    'material' — use Θ from physical parameters ρ, E (requires
                                 CABLE_PROPERTIES to have rho and E filled in).
    verbose       : print per-frequency QC progress.
    """
    from .analysis import compute_mean_eta

    if bw_frac is None:
        from .config import BANDWIDTH_FRAC
        bw_frac = BANDWIDTH_FRAC

    scale = 100.0 if percent else 1.0
    method_label = ('Fourier gradient (∫ε ds)' if strain_method == 'fourier'
                    else 'gradient (∫ε ds)' if strain_method == 'gradient'
                    else 'arc-length (Σδd)')
    est_label = 'envelope |H(·)|' if estimator == 'envelope' else 'instantaneous'
    ylabel = r'Mean coupling efficiency $\eta$ [%]' if percent else r'Mean coupling efficiency $\eta$'

    fig, ax = plt.subplots(figsize=(10, 6.5))

    _eta_theta_theory(ax, percent=percent, show_theory=show_theory)

    theta_key = 'theta_material' if theta_source == 'material' else 'theta_pred'

    # Data points
    plotted = 0
    for cfg in datasets:
        if theta_key not in cfg or cfg[theta_key] is None:
            continue

        # Retrieve or compute mean eta
        key = (f_min, f_max, ref, strain_method, estimator)
        summary = cfg.get('eta_summary', {})
        if key in summary:
            d = summary[key]
            eta_mean, eta_std = d['eta_mean'], d['eta_std']
        else:
            eta_mean, eta_std, _ = compute_mean_eta(
                cfg, f_min=f_min, f_max=f_max, ref=ref,
                bw_frac=bw_frac, grad_direction=grad_direction,
                strain_method=strain_method, estimator=estimator,
                verbose=verbose)

        if eta_mean is None:
            continue

        col, marker, fillstyle = dataset_style(cfg)
        theta = cfg[theta_key]
        theta_std = cfg.get('theta_material_std') if theta_source == 'material' else None
        is_sag = (fillstyle == 'none')

        mew = 1.5 if is_sag else 0.7
        mec = col if is_sag else 'black'

        ax.plot(theta, eta_mean * scale,
                marker=marker, ls='none', color=col,
                fillstyle=fillstyle, ms=13, alpha=0.75,
                markeredgecolor=mec, markeredgewidth=mew, zorder=5)

        xerr = theta_std if (theta_std and theta_std > 0) else None
        yerr = eta_std * scale if (show_errorbars and eta_std > 0) else None
        if xerr is not None or yerr is not None:
            ax.errorbar(theta, eta_mean * scale,
                        xerr=xerr, yerr=yerr,
                        fmt='none', ecolor=col, elinewidth=1.3,
                        capsize=5, capthick=1.3, zorder=4)
        plotted += 1

    _eta_theta_legend(ax)

    ax.set_xscale('log')
    if theta_source == 'material':
        xlabel = r'$\Theta_\mathrm{mat} = \rho^2 g^2 A^3 L^8 / (128\pi^8 E^2 I^3)$  [—]'
    else:
        xlabel = r'$\Theta_\mathrm{sag} = A \cdot w_0^2 / (8I)$  [—]'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(-8 if percent else -0.08, 115 if percent else 1.15)

    ref_lbl = 'shaker δL' if ref == 'shaker' else 'end-sensor Δu'
    src_lbl = 'material (ρ,E)' if theta_source == 'material' else 'sag-based'
    ax.set_title(
        rf'$\eta$ vs $\Theta$ [{src_lbl}]: measured vs. theory'
        f'\nMean η over {f_min:.0f}–{f_max:.0f} Hz  [{ref_lbl},  strain: {method_label},  estimator: {est_label}]'
        f'  (n={plotted} datasets)',
        fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_eta_vs_theta_fd(datasets, f_min=0.0, f_max=50.0, coh_thresh=0.7,
                         theta_source='sag', percent=True,
                         xlim=None, ylim=None,
                         show_errorbars=True, show_theory=True,
                         x_source='ends', verbose=False):
    """η vs Θ scatter using the Welch transfer-function (FRF) method.

    Same styling as plot_eta_vs_theta (colour = cable, shape = gap, open = Sag,
    theory curve η = 1/(1+Θ)), but η is the band-averaged strain-transfer
    amplitude from the FRF: η = ⟨|H_η(f)|⟩ over [f_min, f_max], using only bins
    with coherence γ²(η) ≥ coh_thresh (see freqdomain.band_mean_eta). The error
    bar is the std of |H_η| across those bins.

    For each dataset the FRF in cfg['fd_tf'] is reused if present (so the δL
    reference chosen upstream is honoured); otherwise it is computed here with
    the given x_source. Geometry must be prepared (theta_pred / theta_material).

    Parameters mirror plot_eta_vs_theta; coh_thresh replaces the bandpass/strain
    options, and x_source selects the δL reference for any FRF computed on the fly.
    """
    from .freqdomain import (band_mean_eta, build_coupling_signals,
                             compute_transfer_functions)

    scale = 100.0 if percent else 1.0
    ylabel = (r'Mean coupling efficiency $\eta$ [%]' if percent
              else r'Mean coupling efficiency $\eta$')
    theta_key = 'theta_material' if theta_source == 'material' else 'theta_pred'

    fig, ax = plt.subplots(figsize=(10, 6.5))
    _eta_theta_theory(ax, percent=percent, show_theory=show_theory)

    plotted = 0
    for cfg in datasets:
        if theta_key not in cfg or cfg[theta_key] is None:
            continue
        if cfg.get('fd_tf') is None:
            build_coupling_signals(cfg, x_source=x_source)
            compute_transfer_functions(cfg)
        eta_mean, eta_std, n_bins = band_mean_eta(
            cfg, f_min=f_min, f_max=f_max, coh_thresh=coh_thresh)
        if eta_mean is None:
            if verbose:
                print(f"  [{cfg.get('label', '?')}] no coherent bins in "
                      f"{f_min:.0f}–{f_max:.0f} Hz — skipped.")
            continue

        col, marker, fillstyle = dataset_style(cfg)
        theta = cfg[theta_key]
        theta_std = (cfg.get('theta_material_std')
                     if theta_source == 'material' else None)
        is_sag = (fillstyle == 'none')
        mew = 1.5 if is_sag else 0.7
        mec = col if is_sag else 'black'

        ax.plot(theta, eta_mean * scale, marker=marker, ls='none', color=col,
                fillstyle=fillstyle, ms=13, alpha=0.75,
                markeredgecolor=mec, markeredgewidth=mew, zorder=5)

        xerr = theta_std if (theta_std and theta_std > 0) else None
        yerr = eta_std * scale if (show_errorbars and eta_std and eta_std > 0) else None
        if xerr is not None or yerr is not None:
            ax.errorbar(theta, eta_mean * scale, xerr=xerr, yerr=yerr,
                        fmt='none', ecolor=col, elinewidth=1.3,
                        capsize=5, capthick=1.3, zorder=4)
        plotted += 1

    _eta_theta_legend(ax)

    ax.set_xscale('log')
    if theta_source == 'material':
        xlabel = r'$\Theta_\mathrm{mat} = \rho^2 g^2 A^3 L^8 / (128\pi^8 E^2 I^3)$  [—]'
    else:
        xlabel = r'$\Theta_\mathrm{sag} = A \cdot w_0^2 / (8I)$  [—]'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(-8 if percent else -0.08, 160 if percent else 1.60)

    src_lbl = 'material (ρ,E)' if theta_source == 'material' else 'sag-based'
    ax.set_title(
        rf'$\eta$ vs $\Theta$ [{src_lbl}]: measured (Welch FRF) vs. theory'
        f'\nMean η = ⟨|H_η|⟩ over {f_min:.0f}–{f_max:.0f} Hz  '
        f'[δL = {x_source},  γ² ≥ {coh_thresh:.2f}]  (n={plotted} datasets)',
        fontsize=11)
    plt.tight_layout()
    plt.show()
    return fig


def plot_spectral_peak_overview(datasets, components=('vx', 'vy', 'vz'),
                                f_max=500.0, n_peaks=4,
                                xscale='linear', yscale='linear',
                                unit_scale=1e3, unit_label='|V| [mm/s]'):
    """Mean-spectrum plots with dominant peak annotations.

    One figure per dataset (or shaker-ref); columns = components.
    Peaks are marked with triangle markers + vertical dashed lines; a text box
    lists (rank, freq [Hz], normalised amplitude) for each peak.

    Parameters
    ----------
    datasets   : a single cfg/ref dict **or** a list thereof.
    components : component keys present in the dicts (e.g. 'vx' or 'sh_vx').
    f_max      : upper frequency limit [Hz].
    n_peaks    : number of dominant peaks to annotate.
    xscale, yscale : 'linear' or 'log'.
    unit_scale : multiplier for the y-axis (default 1e3 → mm/s).
    unit_label : y-axis label.
    """
    from .analysis import spectral_peak_summary

    if isinstance(datasets, dict):
        datasets = [datasets]

    for cfg in datasets:
        label = cfg.get('label', cfg.get('path', '<unnamed>'))
        has_style = 'gap_m' in cfg
        col_ds = dataset_style(cfg)[0] if has_style else '#444444'

        summary = spectral_peak_summary(cfg, components=components,
                                        f_max=f_max, n_peaks=n_peaks)

        n_cols = len(components)
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(4.5 * n_cols, 3.8),
                                 squeeze=False)
        axes = axes[0]

        for j, comp in enumerate(components):
            ax = axes[j]
            d = summary[comp]
            freqs, mean_spec, peaks = d['freqs'], d['mean_spec'], d['peaks']

            band_mask = freqs <= f_max
            s_peak = mean_spec[band_mask].max() + 1e-30
            spec_disp = mean_spec * unit_scale

            comp_col = _COMPONENT_COLORS.get(comp, f'C{j}')
            ax.plot(freqs[band_mask], spec_disp[band_mask],
                    lw=1.0, color=comp_col)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel(unit_label if j == 0 else '')
            disp_name = comp.replace('sh_', '')
            ax.set_title(_COMPONENT_LABELS.get(disp_name, comp))
            ax.set_xlim(0, f_max)
            if yscale == 'linear':
                ax.set_ylim(bottom=0)

            box_lines = []
            for rank, (fp, ap) in enumerate(peaks, 1):
                amp_disp = ap * s_peak * unit_scale
                ax.axvline(fp, color=comp_col, lw=0.9, ls='--', alpha=0.6)
                ax.plot(fp, amp_disp, 'v', color=comp_col, ms=7, zorder=5)
                box_lines.append(f'#{rank}  {fp:6.1f} Hz   {ap:.3f}')

            ax.text(0.97, 0.97, '\n'.join(box_lines),
                    transform=ax.transAxes, fontsize=7.5,
                    va='top', ha='right', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white',
                              alpha=0.88, edgecolor='gray', lw=0.8))

        sfx = ' (shaker)' if components[0].startswith('sh_') else ''
        fig.suptitle(f"Mean spectra & dominant peaks — {label}{sfx}", fontsize=12)
        plt.tight_layout()
        plt.show()


def spectral_peak_table(datasets, components=('vx', 'vy', 'vz'),
                        f_max=500.0, n_peaks=4):
    """Build a tidy DataFrame of dominant spectral peaks and display it.

    Columns: Dataset, Component, Rank, Freq_Hz, Norm_Amp.
    Also returns a wide pivot table (one row per Dataset+Component,
    columns = Peak1_Hz, Peak1_Amp, Peak2_Hz, Peak2_Amp, …).

    Both the tidy and wide forms are displayed (IPython rich display when
    available, plain print otherwise).

    Returns the wide-form DataFrame.
    """
    import pandas as pd
    from .analysis import spectral_peak_summary

    if isinstance(datasets, dict):
        datasets = [datasets]

    rows = []
    for cfg in datasets:
        label = cfg.get('label', cfg.get('path', '<unnamed>'))
        sfx = ' (shaker)' if components[0].startswith('sh_') else ''
        summary = spectral_peak_summary(cfg, components=components,
                                        f_max=f_max, n_peaks=n_peaks)
        for comp in components:
            disp_comp = comp.replace('sh_', '')
            for rank, (fp, ap) in enumerate(summary[comp]['peaks'], 1):
                rows.append({
                    'Dataset': label + sfx,
                    'Component': disp_comp,
                    'Rank': rank,
                    'Freq_Hz': round(fp, 1),
                    'Norm_Amp': round(ap, 4),
                })

    df_long = pd.DataFrame(rows)

    # Build wide-form pivot
    wide_cols = {}
    for r in range(1, n_peaks + 1):
        wide_cols[f'Peak{r}_Hz'] = df_long[df_long['Rank'] == r].set_index(
            ['Dataset', 'Component'])['Freq_Hz']
        wide_cols[f'Peak{r}_Amp'] = df_long[df_long['Rank'] == r].set_index(
            ['Dataset', 'Component'])['Norm_Amp']

    idx = df_long[['Dataset', 'Component']].drop_duplicates()
    idx = idx.set_index(['Dataset', 'Component']).index
    wide_data = {}
    for col, series in wide_cols.items():
        wide_data[col] = series.reindex(idx).values
    df_wide = pd.DataFrame(wide_data, index=idx).reset_index()

    try:
        from IPython.display import display as _display
        print(f"\nDominant spectral peaks (top {n_peaks}, 0–{f_max:.0f} Hz)")
        _display(df_wide.style
                 .format({c: '{:.1f}' for c in df_wide.columns if c.endswith('_Hz')})
                 .format({c: '{:.4f}' for c in df_wide.columns if c.endswith('_Amp')})
                 .set_caption(f'Norm. amplitude = fraction of in-band peak'))
    except Exception:
        print(df_wide.to_string(index=False))

    return df_wide


def plot_linearity_eta_time(cfg, results, frequencies=None,
                            smooth_cycles=5, ref='shaker'):
    """η(t) grid: one column per frequency, two rows (η top, |δL| bottom).

    The shaded grey region marks the fade interval at the end of each segment.
    The bold line shows η smoothed with a running median of smooth_cycles cycles.

    Parameters
    ----------
    cfg         : loaded linearity cfg dict (must have eta_pred).
    results     : dict of {f_target: linearity_qc output}, e.g. cfg['lin_results'].
    frequencies : subset of frequencies to plot; default = sorted(results.keys()).
    smooth_cycles : size of the running-median window in cycles of f_target.
    ref         : 'shaker' or 'ends' — which η series to use.
    """
    from scipy.ndimage import median_filter

    freqs = sorted(results.keys()) if frequencies is None else list(frequencies)
    freqs = [f for f in freqs if f in results]
    if not freqs:
        print("No results to plot.")
        return

    col, _, _fs = dataset_style(cfg) if 'gap_m' in cfg else ('#555', 'o', 'full')
    if ref == 'ends':
        eta_key, eta_med_key = 'eta_t_ends', 'eta_med_ends'
    elif ref == 'shaker_end':
        eta_key, eta_med_key = 'eta_t_shend', 'eta_med_shend'
    else:
        eta_key, eta_med_key = 'eta_t', 'eta_med'
    eta_pred = cfg.get('eta_pred', 1.0)

    ncols = len(freqs)
    fig, axes = plt.subplots(2, ncols, figsize=(2.8 * ncols, 6),
                              sharex=False, squeeze=False)

    for j, f in enumerate(freqs):
        res = results[f]
        t_rel = res['t_win'] - res['t_start']
        eta_raw = res[eta_key]
        amp_um = res['amp_env'] * 1e6

        win = max(int(smooth_cycles / f * cfg['fs']), 5)
        eta_fill = np.where(np.isnan(eta_raw), 0.0, eta_raw)
        eta_smooth = median_filter(eta_fill, size=win).astype(float)
        eta_smooth[np.isnan(eta_raw)] = np.nan

        t_ramp_end = res['t_ramp_end']
        ax_top = axes[0, j]
        ax_bot = axes[1, j]

        ax_top.axvspan(t_ramp_end, t_rel[-1] + 0.01, alpha=0.13, color='gray')
        ax_top.axhline(1.0, color='gray', ls='--', lw=0.7)
        ax_top.axhline(eta_pred, color='k', ls=':', lw=0.9,
                       label=f'η_pred={eta_pred:.3f}')
        ax_top.plot(t_rel, eta_raw, color=col, lw=0.3, alpha=0.35)
        ax_top.plot(t_rel, eta_smooth, color=col, lw=1.6)
        ax_top.set_ylim(-0.3, 2.3)
        ax_top.set_title(f'{f} Hz\nη_med={res[eta_med_key]:.3f}', fontsize=9)
        if j == 0:
            ax_top.set_ylabel('η = δxₗ / δL')
        if j == ncols - 1:
            ax_top.legend(fontsize=7, loc='upper right')

        ax_bot.axvspan(t_ramp_end, t_rel[-1] + 0.01, alpha=0.13, color='gray')
        ax_bot.plot(t_rel, amp_um, color='steelblue', lw=1.0)
        ax_bot.set_xlabel('Time in segment [s]')
        if j == 0:
            ax_bot.set_ylabel('|δL| [μm]')
        ax_bot.set_title(f'|δL| pk={amp_um.max():.2f} μm', fontsize=9)

    label = cfg.get('label', '<unnamed>')
    ref_lbl = 'δL_ends' if ref == 'ends' else 'δL_shaker'
    fig.suptitle(f'Linearity QC: η(t)  [{ref_lbl}] — {label}  '
                 f'(grey = fade)', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_linearity_eta_vs_amplitude(datasets, frequencies=None,
                                     smooth_cycles=5, exclude_fade=True,
                                     n_bins=25, ref='shaker'):
    """η vs |δL| amplitude — one figure per dataset, one line per frequency.

    Left panel: raw scatter (thin, transparent). Right panel: binned-median
    line, which gives a clean linearity check: a horizontal line = linear.

    Parameters
    ----------
    datasets      : single cfg dict or list. Each must have cfg['lin_results'].
    frequencies   : subset to plot; default = all in lin_results.
    smooth_cycles : running-median window for the scatter (cycles of f_target).
    exclude_fade  : if True, only use the ramp portion (before fade starts).
    n_bins        : number of amplitude bins for the binned-median panel.
    ref           : 'shaker' or 'ends'.
    """
    from scipy.ndimage import median_filter

    if isinstance(datasets, dict):
        datasets = [datasets]

    if ref == 'ends':
        eta_key = 'eta_t_ends'
    elif ref == 'shaker_end':
        eta_key = 'eta_t_shend'
    else:
        eta_key = 'eta_t'
    cmap = plt.get_cmap('rainbow')

    for cfg in datasets:
        results = cfg.get('lin_results', {})
        freqs = sorted(results.keys()) if frequencies is None else list(frequencies)
        freqs = [f for f in freqs if f in results]
        if not freqs:
            continue

        colors = [cmap(i / max(len(freqs) - 1, 1)) for i in range(len(freqs))]
        eta_pred = cfg.get('eta_pred', 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax_sc, ax_bin = axes

        for fc, f in zip(colors, freqs):
            res = results[f]
            t_rel = res['t_win'] - res['t_start']
            amp_um = res['amp_env'] * 1e6
            eta_raw = res[eta_key].copy()

            win = max(int(smooth_cycles / f * cfg['fs']), 5)
            eta_fill = np.where(np.isnan(eta_raw), 0.0, eta_raw)
            eta_smooth = median_filter(eta_fill, size=win).astype(float)
            eta_smooth[np.isnan(eta_raw)] = np.nan

            if exclude_fade:
                sel = res['ramp_mask'].copy()
            else:
                sel = np.ones(len(t_rel), dtype=bool)

            amp_s, eta_s = amp_um[sel], eta_smooth[sel]
            valid = ~np.isnan(eta_s) & (amp_s > 0)
            if valid.sum() < 10:
                continue

            ax_sc.scatter(amp_s[valid], eta_s[valid], s=3, color=fc, alpha=0.25,
                          linewidths=0)

            amp_v, eta_v = amp_s[valid], eta_s[valid]
            edges = np.linspace(amp_v.min(), amp_v.max(), n_bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            bin_med = [np.nanmedian(eta_v[(amp_v >= lo) & (amp_v < hi)])
                       if ((amp_v >= lo) & (amp_v < hi)).sum() > 3 else np.nan
                       for lo, hi in zip(edges[:-1], edges[1:])]
            ax_bin.plot(centers, bin_med, 'o-', color=fc, ms=5, lw=1.5,
                        label=f'{f} Hz')

        for ax in axes:
            ax.axhline(1.0, color='gray', ls='--', lw=0.8)
            ax.axhline(eta_pred, color='k', ls=':', lw=0.9,
                       label=f'η_pred={eta_pred:.3f}')
            ax.set_xlabel('|δL| [μm]')
            ax.set_ylabel('η = δxₗ / δL')
            ax.set_ylim(-0.2, 2.2)

        ax_sc.set_title('Scatter (running-median smoothed η)')
        ax_bin.set_title(f'Binned median η  ({n_bins} amplitude bins)')
        ax_bin.legend(fontsize=8, ncol=2)

        ref_lbl = 'δL_ends' if ref == 'ends' else 'δL_shaker'
        fade_lbl = ' — ramp only' if exclude_fade else ''
        label = cfg.get('label', '<unnamed>')
        fig.suptitle(f'Amplitude linearity check [{ref_lbl}]{fade_lbl} — {label}',
                     fontsize=11)
        plt.tight_layout()
        plt.show()


def plot_linearity_comparison(datasets, f_target, smooth_cycles=5,
                               exclude_fade=True, ref='shaker', n_bins=25):
    """Compare η vs |δL| across multiple cables for a single frequency.

    One figure; one line per dataset in the binned-median panel, plus a scatter
    panel with all datasets overlaid.
    """
    from scipy.ndimage import median_filter

    if isinstance(datasets, dict):
        datasets = [datasets]

    if ref == 'ends':
        eta_key = 'eta_t_ends'
    elif ref == 'shaker_end':
        eta_key = 'eta_t_shend'
    else:
        eta_key = 'eta_t'

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_sc, ax_bin = axes

    for k, cfg in enumerate(datasets):
        results = cfg.get('lin_results', {})
        if f_target not in results:
            continue
        res = results[f_target]
        col, _, _fs = dataset_style(cfg) if 'gap_m' in cfg else (f'C{k}', 'o', 'full')
        lbl = cfg.get('label', f'Dataset {k}')

        t_rel = res['t_win'] - res['t_start']
        amp_um = res['amp_env'] * 1e6
        eta_raw = res[eta_key].copy()

        win = max(int(smooth_cycles / f_target * cfg['fs']), 5)
        eta_fill = np.where(np.isnan(eta_raw), 0.0, eta_raw)
        eta_smooth = median_filter(eta_fill, size=win).astype(float)
        eta_smooth[np.isnan(eta_raw)] = np.nan

        sel = res['ramp_mask'] if exclude_fade else np.ones(len(t_rel), dtype=bool)
        amp_s, eta_s = amp_um[sel], eta_smooth[sel]
        valid = ~np.isnan(eta_s) & (amp_s > 0)
        if valid.sum() < 5:
            continue

        ax_sc.scatter(amp_s[valid], eta_s[valid], s=3, color=col, alpha=0.2,
                      linewidths=0)

        amp_v, eta_v = amp_s[valid], eta_s[valid]
        edges = np.linspace(amp_v.min(), amp_v.max(), n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_med = [np.nanmedian(eta_v[(amp_v >= lo) & (amp_v < hi)])
                   if ((amp_v >= lo) & (amp_v < hi)).sum() > 3 else np.nan
                   for lo, hi in zip(edges[:-1], edges[1:])]
        ax_bin.plot(centers, bin_med, 'o-', color=col, ms=5, lw=1.5, label=lbl)

        ax_bin.axhline(cfg.get('eta_pred', 1.0), color=col, ls=':', lw=0.7, alpha=0.6)

    for ax in axes:
        ax.axhline(1.0, color='gray', ls='--', lw=0.8, label='η = 1')
        ax.set_xlabel('|δL| [μm]')
        ax.set_ylabel('η = δxₗ / δL')
        ax.set_ylim(-0.2, 2.2)

    ax_sc.set_title('Scatter (all cables)')
    ax_bin.set_title(f'Binned median η per cable  ({n_bins} bins)')
    ax_bin.legend(fontsize=8)

    ref_lbl = 'δL_ends' if ref == 'ends' else 'δL_shaker'
    fade_lbl = ' — ramp only' if exclude_fade else ''
    fig.suptitle(f'{f_target} Hz — cross-cable comparison [{ref_lbl}]{fade_lbl}',
                 fontsize=11)
    plt.tight_layout()
    plt.show()


def _select_1d_trace(source, comp, sensor_index=None):
    """Return a 1-D trace from `source[comp]`.

    `source` may be a shaker-reference dict (1-D component arrays) or a cable
    cfg (2-D arrays of shape (N_t, N_sensors)). For 2-D arrays a sensor_index
    is required (raises ValueError if missing or out of range).
    """
    x = np.asarray(source[comp])
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"Component '{comp}' has unexpected shape {x.shape}")
    n_sensors = x.shape[1]
    if sensor_index is None:
        raise ValueError(
            f"Component '{comp}' is 2-D (N_t={x.shape[0]}, N_sensors={n_sensors}); "
            f"pass sensor_index=0..{n_sensors-1} to pick a sensor."
        )
    if not 0 <= sensor_index < n_sensors:
        raise ValueError(
            f"sensor_index={sensor_index} out of range [0, {n_sensors-1}] for '{comp}'."
        )
    return x[:, sensor_index]


def plot_spectrogram_grid(source, components=('vx', 'vy', 'vz'),
                          sensor_index=None,
                          nperseg=None, noverlap=None,
                          f_max=None, db_floor=-80, title=None,
                          ax_height=2.6):
    """Stacked spectrograms (one row per component).

    Works for any dict that exposes `'fs'` plus component arrays:
      * Shaker-reference dicts (1-D per component) — `sensor_index` ignored.
      * Cable cfg dicts (2-D (N_t, N_sensors) per component) — pass
        `sensor_index` to pick one LDV sensor.

    Power is shown in dB relative to that panel's own peak so dynamic range
    is comparable across components. `db_floor` sets the colour-scale floor.
    """
    fs = source['fs']
    n = len(components)
    fig, axes = plt.subplots(n, 1, figsize=(11, ax_height * n),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for i, comp in enumerate(components):
        ax = axes[i]
        x = _select_1d_trace(source, comp, sensor_index=sensor_index)
        f, tt, Sxx = compute_spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap,
                                          mode='magnitude')
        P = Sxx ** 2
        P_norm = P / (P.max() + 1e-30)
        Sdb = 10.0 * np.log10(P_norm + 1e-30)
        pcm = ax.pcolormesh(tt, f, Sdb, cmap='viridis',
                            vmin=db_floor, vmax=0.0, shading='auto')
        ax.set_ylabel(f"{_COMPONENT_LABELS.get(comp, comp)}\nFrequency [Hz]")
        if f_max is not None:
            ax.set_ylim(top=f_max)
        plt.colorbar(pcm, ax=ax, label='dB (vs peak)', shrink=0.85)

    axes[-1].set_xlabel('Time [s]')
    if title is None:
        title = source.get('label', source.get('path', 'spectrogram'))
    suffix = f"  (sensor {sensor_index})" if sensor_index is not None else ""
    fig.suptitle(f"Spectrogram — {title}{suffix}", fontsize=12, y=1.0)
    plt.tight_layout()
    plt.show()


def plot_fk_spectrum(cfg, components=('vx', 'vy', 'vz'),
                    f_max=500.0, db_floor=-40.0,
                    title=None, ax_height=3.5):
    """F-K (frequency-wavenumber) spectrum for each component.

    Requires a cable cfg dict with 2-D velocity arrays (N_t, N_sensors) and
    `cfg['x']` sensor positions [m]. One panel per component, stacked vertically.

    Parameters
    ----------
    cfg       : loaded cable cfg dict.
    components: component keys (only 2-D arrays are valid).
    f_max     : upper frequency limit for the y-axis [Hz].
    db_floor  : colour-scale minimum [dB relative to peak].
    title     : figure title override.
    ax_height : height per subplot [inches].
    """
    x = cfg['x']
    n = len(components)
    fig, axes = plt.subplots(n, 1, figsize=(9, ax_height * n),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    dx_used = None

    for i, comp in enumerate(components):
        ax = axes[i]
        arr = cfg.get(comp)
        if arr is None or np.asarray(arr).ndim != 2:
            ax.text(0.5, 0.5, f'{comp}: no 2-D data',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        freqs, k, FK_db, dx = compute_fk_spectrum(arr, cfg['fs'], x)
        if dx_used is None:
            dx_used = dx

        f_mask = freqs <= f_max
        pcm = ax.pcolormesh(k, freqs[f_mask], FK_db[f_mask, :],
                            cmap='inferno', vmin=db_floor, vmax=0.0,
                            shading='auto')
        plt.colorbar(pcm, ax=ax, label='dB (vs peak)', shrink=0.85)
        ax.axvline(0, color='white', lw=0.6, ls='--', alpha=0.4)
        ax.set_ylabel(f"{_COMPONENT_LABELS.get(comp, comp)}\nFrequency [Hz]")

    axes[-1].set_xlabel('Wavenumber [1/m]')
    lbl = title or cfg.get('label', '')
    dx_str = f'  (dx ≈ {dx_used*1e3:.1f} mm)' if dx_used is not None else ''
    fig.suptitle(f"F-K spectrum — {lbl}{dx_str}", fontsize=12, y=1.0)
    plt.tight_layout()
    plt.show()


def plot_theta_comparison(datasets, annotate=True, show_theory_line=True,
                           xlim=None, ylim=None):
    """Compare sag-based and material-based Θ values across all datasets.

    Produces two panels:
      Left  — scatter Θ_sag vs Θ_material with the 1:1 line.
               Each point represents one dataset; colour = cable, marker = gap.
               Points only appear once ρ and E are filled in CABLE_PROPERTIES.
      Right — bar chart of Θ_sag and Θ_material side-by-side per dataset,
               so you can see relative agreement at a glance.

    Requires prepare_geometry (or plot_initial_geometry) to have been called
    so that cfg['theta_pred'] and cfg['theta_material'] are populated.

    Parameters
    ----------
    datasets         : list of loaded cable cfg dicts with geometry computed.
    annotate         : label each scatter point with the dataset label.
    show_theory_line : draw the 1:1 reference line on the scatter panel.
    xlim, ylim       : axis limits for the scatter panel.
    """
    from .config import CABLE_COLORS, GAP_MARKERS
    from matplotlib.lines import Line2D

    valid = [cfg for cfg in datasets
             if cfg.get('theta_pred') is not None
             and cfg.get('theta_material') is not None]
    all_sag = [cfg for cfg in datasets if cfg.get('theta_pred') is not None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_sc, ax_bar = axes

    # ── Left: scatter Θ_sag vs Θ_material ───────────────────────────────────
    if show_theory_line and valid:
        vals = [cfg['theta_pred'] for cfg in valid] + [cfg['theta_material'] for cfg in valid]
        lo, hi = min(vals), max(vals)
        margin = 0.5
        rng = np.logspace(np.log10(lo) - margin, np.log10(hi) + margin, 100)
        ax_sc.plot(rng, rng, 'k--', lw=1.2, alpha=0.6, label='1:1 line')

    for cfg in valid:
        col, marker, fillstyle = dataset_style(cfg)
        th_mat = cfg['theta_material']
        th_std = cfg.get('theta_material_std') or 0.0
        ax_sc.scatter(cfg['theta_pred'], th_mat,
                      c=col, marker=marker, s=120, alpha=0.85,
                      edgecolors='black' if fillstyle == 'full' else col,
                      linewidths=1.5 if fillstyle == 'none' else 0.7,
                      zorder=5)
        if th_std > 0:
            ax_sc.errorbar(cfg['theta_pred'], th_mat,
                           yerr=th_std,
                           fmt='none', ecolor=col, elinewidth=1.2,
                           capsize=4, capthick=1.2, zorder=4)
        if annotate:
            ax_sc.annotate(cfg['label'],
                           (cfg['theta_pred'], th_mat),
                           fontsize=6.5, textcoords='offset points',
                           xytext=(4, 3), alpha=0.85)

    ax_sc.set_xscale('log')
    ax_sc.set_yscale('log')
    ax_sc.set_xlabel(r'$\Theta_\mathrm{sag} = A \cdot w_0^2 / (8I)$', fontsize=11)
    ax_sc.set_ylabel(r'$\Theta_\mathrm{mat} = \rho^2 g^2 A^3 L^8 / (128\pi^8 E^2 I^3)$',
                     fontsize=11)
    if xlim:
        ax_sc.set_xlim(xlim)
    if ylim:
        ax_sc.set_ylim(ylim)
    if show_theory_line and valid:
        ax_sc.legend(fontsize=9)
    n_missing = len(all_sag) - len(valid)
    title_sc = r'$\Theta_\mathrm{sag}$ vs $\Theta_\mathrm{mat}$'
    if n_missing:
        title_sc += f'\n({n_missing} datasets hidden — ρ/E not yet set)'
    ax_sc.set_title(title_sc, fontsize=11)

    # ── Right: bar chart Θ_sag and Θ_material per dataset ───────────────────
    labels = [cfg['label'] for cfg in all_sag]
    theta_sag_vals = [cfg['theta_pred'] for cfg in all_sag]
    theta_mat_vals = [cfg.get('theta_material') for cfg in all_sag]

    x_pos = np.arange(len(labels))
    width = 0.38
    bar_colors = [dataset_style(cfg)[0] for cfg in all_sag]

    bars_sag = ax_bar.bar(x_pos - width / 2, theta_sag_vals, width,
                          color=bar_colors, alpha=0.7, label=r'$\Theta_\mathrm{sag}$',
                          edgecolor='black', linewidth=0.5)

    theta_mat_std_vals = [cfg.get('theta_material_std') or 0.0 for cfg in all_sag]
    mat_present = [v for v in theta_mat_vals if v is not None]
    if mat_present:
        mat_plot = [v if v is not None else 0.0 for v in theta_mat_vals]
        mat_err = [s if theta_mat_vals[i] is not None else 0.0
                   for i, s in enumerate(theta_mat_std_vals)]
        bars_mat = ax_bar.bar(x_pos + width / 2, mat_plot, width,
                              color=bar_colors, alpha=0.4, hatch='//',
                              label=r'$\Theta_\mathrm{mat}$',
                              edgecolor='black', linewidth=0.5)
        ax_bar.errorbar(x_pos + width / 2, mat_plot, yerr=mat_err,
                        fmt='none', ecolor='black', elinewidth=1.0,
                        capsize=3, capthick=1.0, zorder=5)

    ax_bar.set_yscale('log')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax_bar.set_ylabel(r'$\Theta$  [—]')
    ax_bar.axhline(1.0 / 32.0, color='gray', ls=':', lw=1.0, alpha=0.7,
                   label=r'$\Theta = 1/32$  (rod-like limit)')
    ax_bar.legend(fontsize=9)
    ax_bar.set_title(r'$\Theta_\mathrm{sag}$ (solid) vs $\Theta_\mathrm{mat}$ (hatched) per dataset',
                     fontsize=11)

    fig.suptitle('Θ comparison: sag-based vs. material-based', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_sag_vs_force_ratio(datasets, g=9.81, annotate=True,
                             xlim=None, ylim=None):
    """Recreate Fig. 2a of Probst et al.: dimensionless sag vs. force ratio.

    Y-axis : w₀(L/2) / L  =  static_sag / chord_len           (dimensionless)
    X-axis : ρ A g L³ / (4π⁴ E I)                             (dimensionless)

    The analytical prediction (Eq. 15) is the 1:1 diagonal line.
    Each dataset with geometry computed AND with ρ and E filled in
    CABLE_PROPERTIES contributes one marker.

    Parameters
    ----------
    datasets : list of cable cfg dicts — geometry must be computed first via
               prepare_geometry (or plot_initial_geometry).
    g        : gravitational acceleration [m/s²] (default 9.81).
    annotate : label each point with the dataset label.
    xlim, ylim : optional axis limits (tuple); if None, auto-ranged.
    """
    from .config import CABLE_PROPERTIES, cable_section_AI

    fig, ax = plt.subplots(figsize=(7, 6))

    points = []
    for cfg in datasets:
        if 'static_sag' not in cfg or 'chord_len' not in cfg:
            continue
        cable_name = cfg.get('cable')
        props = CABLE_PROPERTIES.get(cable_name, {})
        rho = props.get('rho')
        E_raw = props.get('E')
        if rho is None or E_raw is None:
            continue

        A, I = cable_section_AI(cable_name)
        L = cfg['chord_len']
        sag = cfg['static_sag']
        C = rho * A * g * L ** 3 / (4 * np.pi ** 4 * I)

        if isinstance(E_raw, (list, tuple)):
            xs = np.asarray([C / e for e in E_raw])
            x, x_std = float(xs.mean()), float(xs.std())
        else:
            x, x_std = float(C / E_raw), 0.0

        y = sag / L
        points.append((x, x_std, y, cfg, cable_name))

    if not points:
        ax.text(0.5, 0.5,
                'No data to plot.\nFill in ρ and E in CABLE_PROPERTIES first.',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Dimensionless sag vs. force ratio (Fig. 2a)')
        plt.tight_layout()
        plt.show()
        return

    # Analytical line  y = x  over the range of the data
    all_x = [p[0] for p in points]
    lo = min(all_x)
    hi = max(all_x)
    pad = 1.5
    line_rng = np.logspace(np.log10(lo) - pad, np.log10(hi) + pad, 300)
    ax.plot(line_rng, line_rng, 'k-', lw=2.0, alpha=0.85, zorder=1,
            label=r'Analytical: $w_0/L = \rho A g L^3 / (4\pi^4 E I)$')

    # Data points
    for x, x_std, y, cfg, cable_name in points:
        col, marker, fillstyle = dataset_style(cfg)
        is_sag = (fillstyle == 'none')
        ax.scatter(x, y,
                   c=col, marker=marker, s=130, alpha=0.85, zorder=5,
                   edgecolors=col if is_sag else 'black',
                   linewidths=1.8 if is_sag else 0.8)
        if x_std > 0:
            ax.errorbar(x, y, xerr=x_std,
                        fmt='none', ecolor=col, elinewidth=1.2,
                        capsize=4, capthick=1.2, zorder=4)
        if annotate:
            ax.annotate(cfg['label'], (x, y), fontsize=6.5,
                        textcoords='offset points', xytext=(5, 3), alpha=0.9)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Force ratio  $\rho A g L^3 \,/\, (4\pi^4 E I)$',
                  fontsize=12)
    ax.set_ylabel(r'Dimensionless sag  $w_0(L/2)\,/\,L$', fontsize=12)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Custom legend: cables + gaps + sag/no-sag
    from .config import CABLE_COLORS, GAP_MARKERS
    from matplotlib.lines import Line2D

    def _mh(marker, color, fillstyle, label, mew=0.8, mec='black', ms=9):
        kw = dict(marker=marker, linewidth=0, markersize=ms,
                  markerfacecolor='none' if fillstyle == 'none' else color,
                  markeredgecolor=color if fillstyle == 'none' else mec,
                  markeredgewidth=1.8 if fillstyle == 'none' else mew,
                  fillstyle=fillstyle, label=label)
        return Line2D([0], [0], **kw)

    sep = Line2D([0], [0], color='none', label=' ')
    cable_handles = [_mh('o', col, 'full', name, mec='black', mew=0.5)
                     for name, col in CABLE_COLORS.items()]
    gap_handles = [_mh(mkr, '#888', 'full', f'{int(g_*100)} cm', mec='black', mew=0.5)
                   for g_, mkr in GAP_MARKERS.items()]
    sag_handles = [_mh('o', '#555', 'full', 'No sag', mec='black', mew=0.5),
                   _mh('o', '#555', 'none', 'Sag', mec='#555', mew=1.8)]

    theory_h = ax.get_legend_handles_labels()[0]
    ax.legend(handles=theory_h + [sep] + cable_handles + [sep] +
              gap_handles + [sep] + sag_handles,
              fontsize=8, loc='upper left', framealpha=0.9)

    n_shown = len(points)
    n_missing = sum(
        1 for cfg in datasets
        if cfg.get('static_sag') is not None
        and (CABLE_PROPERTIES.get(cfg.get('cable', ''), {}).get('rho') is None
             or CABLE_PROPERTIES.get(cfg.get('cable', ''), {}).get('E') is None)
    )
    subtitle = f'n = {n_shown} datasets'
    if n_missing:
        subtitle += f'  ({n_missing} hidden — ρ or E not yet set)'
    ax.set_title('Dimensionless sag vs. force ratio  (Fig. 2a)\n' + subtitle,
                 fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_sag_comparison(datasets, g=9.81, unit='mm', annotate=True,
                        exclude_sag_variants=False,
                        ldv_resolution_m=200e-6):
    """Compare measured midpoint sag with the tension-free theoretical prediction.

    Theoretical sag (Eq. 8 / 15 of Probst et al.):
        w₀_theory = ρ A g L⁴ / (4π⁴ E I)

    This is the analytical prediction for a cable with no initial axial tension.
    Real cables under pre-tension will sag less, so measured sag can be lower.
    Conversely, if the cable has an inherent bend from being on a spool, the
    measured sag may exceed the theory.

    Two panels are produced:
      Left  — scatter: measured sag (x-axis) vs theoretical sag (y-axis) with the
               1:1 line. Points that fall on the line confirm the model; points
               below the line indicate the cable sags less than predicted (pre-tension).
      Right — grouped bar chart per dataset: solid bar = measured, hatched bar =
               theoretical (±std from spread of E measurements), on a log scale so
               cables with very different sag magnitudes are visible simultaneously.

    Parameters
    ----------
    datasets            : list of cable cfg dicts — geometry must be computed first via
                          prepare_geometry (or plot_initial_geometry).
    g                   : gravitational acceleration [m/s²] (default 9.81).
    unit                : 'mm' (default) or 'm' — display unit for sag values.
    annotate            : label each scatter point with the dataset name.
    exclude_sag_variants: if True, skip datasets whose label contains 'Sag'
                          (i.e. intentionally induced-sag configurations).
    ldv_resolution_m    : LDV geometry resolution limit [m] drawn as a dotted line
                          on both panels (default 200 µm).  Pass None to disable.
    """
    from .config import CABLE_PROPERTIES, cable_section_AI, CABLE_COLORS, GAP_MARKERS
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    scale = 1e3 if unit == 'mm' else 1.0
    unit_lbl = 'mm' if unit == 'mm' else 'm'

    records = []
    for cfg in datasets:
        if 'static_sag' not in cfg or 'chord_len' not in cfg:
            continue
        if exclude_sag_variants and 'Sag' in cfg.get('label', ''):
            continue
        cable_name = cfg.get('cable')
        props = CABLE_PROPERTIES.get(cable_name, {})
        rho = props.get('rho')
        E_raw = props.get('E')
        if rho is None or E_raw is None:
            continue

        A, I = cable_section_AI(cable_name)
        L = cfg['chord_len']
        C = rho * A * g * L ** 4 / (4 * np.pi ** 4 * I)

        if isinstance(E_raw, (list, tuple)):
            w_arr = np.asarray([C / e for e in E_raw])
            w_theory = float(w_arr.mean())
            w_theory_std = float(w_arr.std())
        else:
            w_theory = float(C / E_raw)
            w_theory_std = 0.0

        records.append(dict(
            label=cfg['label'],
            cfg=cfg,
            cable_name=cable_name,
            w_meas=cfg['static_sag'],
            w_theory=w_theory,
            w_theory_std=w_theory_std,
        ))

    fig, (ax_sc, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: scatter measured vs theoretical ────────────────────────────────
    if records:
        all_vals = ([r['w_meas'] for r in records]
                    + [r['w_theory'] for r in records])
        lo = min(v for v in all_vals if v > 0)
        hi = max(all_vals)
        pad = 0.5
        rng = np.logspace(np.log10(lo) - pad, np.log10(hi) + pad, 200)
        ax_sc.plot(rng * scale, rng * scale, 'k--', lw=1.5, alpha=0.7,
                   label='1:1  (theory = measurement)', zorder=1)

    for r in records:
        col, marker, fillstyle = dataset_style(r['cfg'])
        is_sag_variant = (fillstyle == 'none')
        x = r['w_meas'] * scale
        y = r['w_theory'] * scale
        y_err = r['w_theory_std'] * scale

        ax_sc.scatter(x, y,
                      c=col, marker=marker, s=120, alpha=0.85, zorder=5,
                      edgecolors=col if is_sag_variant else 'black',
                      linewidths=1.8 if is_sag_variant else 0.8)
        if y_err > 0:
            ax_sc.errorbar(x, y, yerr=y_err,
                           fmt='none', ecolor=col, elinewidth=1.2,
                           capsize=4, capthick=1.2, zorder=4)
        if annotate:
            ax_sc.annotate(r['label'], (x, y), fontsize=6.5,
                           textcoords='offset points', xytext=(4, 3), alpha=0.9)

    ax_sc.set_xscale('log')
    ax_sc.set_yscale('log')
    ax_sc.set_xlabel(f'Measured sag  $w_0$ [{unit_lbl}]', fontsize=12)
    ax_sc.set_ylabel(f'Theoretical sag  $\\rho A g L^4/(4\\pi^4 E I)$ [{unit_lbl}]',
                     fontsize=12)
    ax_sc.set_title('Measured vs. theoretical sag\n(error bars = std from E measurements)',
                    fontsize=11)

    if ldv_resolution_m is not None:
        res_scaled = ldv_resolution_m * scale
        res_lbl = f'LDV res. ({ldv_resolution_m * 1e6:.0f} µm)'
        ax_sc.axvline(res_scaled, color='red', linestyle=':', lw=1.5,
                      alpha=0.8, label=res_lbl, zorder=2)
        ax_sc.axhline(res_scaled, color='red', linestyle=':', lw=1.5,
                      alpha=0.8, zorder=2)

    # Legend for scatter
    def _mh(mkr, col, fs, lbl, mew=0.8, mec='black', ms=9):
        return Line2D([0], [0], marker=mkr, linewidth=0, markersize=ms,
                      markerfacecolor='none' if fs == 'none' else col,
                      markeredgecolor=col if fs == 'none' else mec,
                      markeredgewidth=1.8 if fs == 'none' else mew,
                      fillstyle=fs, label=lbl)

    sep = Line2D([0], [0], color='none', label=' ')
    cable_h = [_mh('o', c, 'full', n, mec='black', mew=0.5)
               for n, c in CABLE_COLORS.items()]
    gap_h = [_mh(mk, '#888', 'full', f'{int(gv*100)} cm', mec='black', mew=0.5)
             for gv, mk in GAP_MARKERS.items()]
    sag_h = [_mh('o', '#555', 'full', 'No sag', mec='black', mew=0.5),
             _mh('o', '#555', 'none', 'Sag variant', mec='#555', mew=1.8)]
    theory_h = ax_sc.get_legend_handles_labels()[0]
    ax_sc.legend(handles=theory_h + [sep] + cable_h + [sep] + gap_h + [sep] + sag_h,
                 fontsize=7.5, loc='upper left', framealpha=0.9)

    # ── Right: grouped bar chart per dataset ─────────────────────────────────
    if not records:
        ax_bar.text(0.5, 0.5, 'No data — fill in ρ and E in CABLE_PROPERTIES.',
                    ha='center', va='center', transform=ax_bar.transAxes, fontsize=11)
    else:
        labels = [r['label'] for r in records]
        w_meas_vals = np.asarray([r['w_meas'] * scale for r in records])
        w_th_vals = np.asarray([r['w_theory'] * scale for r in records])
        w_th_errs = np.asarray([r['w_theory_std'] * scale for r in records])
        bar_colors = [dataset_style(r['cfg'])[0] for r in records]

        x_pos = np.arange(len(labels))
        width = 0.38

        ax_bar.bar(x_pos - width / 2, w_meas_vals, width,
                   color=bar_colors, alpha=0.85,
                   edgecolor='black', linewidth=0.5,
                   label='Measured $w_0$')
        ax_bar.bar(x_pos + width / 2, w_th_vals, width,
                   color=bar_colors, alpha=0.4, hatch='//',
                   edgecolor='black', linewidth=0.5,
                   label='Theoretical $w_0$  (tension-free)')
        ax_bar.errorbar(x_pos + width / 2, w_th_vals, yerr=w_th_errs,
                        fmt='none', ecolor='black', elinewidth=1.0,
                        capsize=3, capthick=1.0, zorder=5)

        ax_bar.set_yscale('log')
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax_bar.set_ylabel(f'Midpoint sag  $w_0$ [{unit_lbl}]', fontsize=12)

        bar_legend_handles = [
            Patch(facecolor='gray', alpha=0.85, edgecolor='black',
                  linewidth=0.5, label='Measured $w_0$'),
            Patch(facecolor='gray', alpha=0.4, hatch='//', edgecolor='black',
                  linewidth=0.5, label='Theoretical $w_0$ (tension-free)'),
        ]
        if ldv_resolution_m is not None:
            res_scaled = ldv_resolution_m * scale
            res_lbl = f'LDV res. ({ldv_resolution_m * 1e6:.0f} µm)'
            ax_bar.axhline(res_scaled, color='red', linestyle=':', lw=1.5,
                           alpha=0.8, zorder=5)
            bar_legend_handles.append(
                Line2D([0], [0], color='red', linestyle=':', lw=1.5, label=res_lbl)
            )
        ax_bar.legend(handles=bar_legend_handles, fontsize=9, loc='best')
        ax_bar.set_title('Measured (solid) vs. theoretical (hatched) sag per dataset\n'
                         '(hatched error bars = std from E measurements)', fontsize=11)

    n_hidden = sum(
        1 for cfg in datasets
        if cfg.get('static_sag') is not None
        and (CABLE_PROPERTIES.get(cfg.get('cable', ''), {}).get('rho') is None
             or CABLE_PROPERTIES.get(cfg.get('cable', ''), {}).get('E') is None)
    )
    suffix = f'  ({n_hidden} datasets hidden — ρ or E not set)' if n_hidden else ''
    fig.suptitle(f'Sag comparison: measured vs. tension-free theory{suffix}',
                 fontsize=13)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain transfer functions — Fig. 3 reproduction
# ─────────────────────────────────────────────────────────────────────────────
def _align_point_phase(pt_freqs, pt_phase_deg, curve_freqs, curve_phase_deg):
    """Shift sparse point phases by ±360° multiples to match the unwrapped curve.

    Keeps STFT cross-check markers on the same branch as the unwrapped Welch
    phase curve so they overlay sensibly.
    """
    if len(curve_freqs) == 0 or len(pt_freqs) == 0:
        return pt_phase_deg
    ref = np.interp(pt_freqs, curve_freqs, curve_phase_deg)
    return pt_phase_deg + 360.0 * np.round((ref - pt_phase_deg) / 360.0)


def plot_fig3_transfer(datasets, f_max=60.0, coh_thresh=0.7,
                       show_stft=True, show_eta_pred=True,
                       figsize=(13, 9), title=None):
    """Reproduce Fig. 3 of Cable_coupling_v1_Simone.pdf from measured FRFs.

    One overlaid four-panel figure across the given datasets:
        (a) |H_mid(f)| normalised to its own max — "normalized midpoint amplitude"
        (b) arg(H_mid(f)) [deg], unwrapped         — "midpoint phase"
        (c) |H_η(f)| × 100 %                        — "strain transfer amplitude"
        (d) arg(H_η(f)) [deg], unwrapped            — "strain transfer phase"

    Frequency axis is linear, 0 to f_max [Hz]. Bins with γ² < coh_thresh are
    greyed out (plotted faint); high-coherence bins are drawn in the cable
    colour. Vertical dotted lines mark the identified resonance f₁. Panel (c)
    overlays the quasi-static prediction η = 1/(1+Θ) as a horizontal dashed line
    per dataset; with a single dataset the amplification region above f₁ is
    shaded.

    Each cfg must carry cfg['fd_tf'] (and cfg['fd_resonance']; cfg['fd_stft']
    if show_stft) — run freqdomain.run_frequency_domain(cfg) first.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    single = len(datasets) == 1

    # Global normalisation for panel (a): divide every dataset by the single
    # largest |H_mid| across all datasets (within the plotted band), so the
    # curves keep their relative amplitudes instead of each peaking at 1.
    norm = 0.0
    for cfg in datasets:
        tf = cfg.get('fd_tf')
        if tf is None:
            continue
        band = (tf['f'] >= 0) & (tf['f'] <= f_max)
        norm = max(norm, float(np.abs(tf['H_mid'][band]).max()))
    if norm <= 0:
        norm = 1.0

    for cfg in datasets:
        tf = cfg.get('fd_tf')
        if tf is None:
            print(f"  [{cfg.get('label', '?')}] no fd_tf — skipping "
                  "(run freqdomain.run_frequency_domain first).")
            continue
        col, marker, _fs = dataset_style(cfg)
        # Legend label with the configuration's Θ value.
        lbl = cfg.get('label')
        if cfg.get('theta_pred') is not None:
            lbl = f"{lbl}  (Θ={cfg['theta_pred']:.3g})"
        f = tf['f']
        band = (f >= 0) & (f <= f_max)
        fb = f[band]
        good_mid = tf['coh_mid'][band] >= coh_thresh
        good_eta = tf['coh_eta'][band] >= coh_thresh

        # Panel a — midpoint amplitude, normalised to the global max
        amp_mid_n = np.abs(tf['H_mid'][band]) / norm
        ax_a.plot(fb, amp_mid_n, color='0.8', lw=0.8, zorder=1)
        ax_a.plot(fb, np.where(good_mid, amp_mid_n, np.nan), color=col, lw=1.6,
                  label=lbl, zorder=2)

        # Panel b — midpoint phase (unwrapped, deg)
        ph_mid = np.rad2deg(np.unwrap(np.angle(tf['H_mid'][band])))
        ax_b.plot(fb, ph_mid, color='0.8', lw=0.8, zorder=1)
        ax_b.plot(fb, np.where(good_mid, ph_mid, np.nan), color=col, lw=1.6,
                  zorder=2)

        # Panel c — strain-transfer amplitude [%]
        amp_eta = np.abs(tf['H_eta'][band]) * 100.0
        ax_c.plot(fb, amp_eta, color='0.8', lw=0.8, zorder=1)
        ax_c.plot(fb, np.where(good_eta, amp_eta, np.nan), color=col, lw=1.6,
                  label=lbl, zorder=2)

        # Panel d — strain-transfer phase (unwrapped, deg)
        ph_eta = np.rad2deg(np.unwrap(np.angle(tf['H_eta'][band])))
        ax_d.plot(fb, ph_eta, color='0.8', lw=0.8, zorder=1)
        ax_d.plot(fb, np.where(good_eta, ph_eta, np.nan), color=col, lw=1.6,
                  zorder=2)

        # Quasi-static η = 1/(1+Θ) on panel c
        if show_eta_pred and cfg.get('eta_pred') is not None:
            ax_c.axhline(cfg['eta_pred'] * 100.0, color=col, ls='--', lw=1.0,
                         alpha=0.6, zorder=1)

        # Resonance marker f₁
        res = cfg.get('fd_resonance')
        f1 = res['f1_meas'] if res else None
        if f1 is not None and f1 <= f_max:
            for ax in (ax_a, ax_b, ax_c, ax_d):
                ax.axvline(f1, color=col, ls=':', lw=1.0, alpha=0.8, zorder=1)
            if single:
                # Θ-dependent amplification region (constructive coupling above f₁)
                ax_c.axvspan(f1, f_max, color='orange', alpha=0.08, zorder=0,
                             label='amplification region (f > f₁)')

        # STFT-per-frequency cross-check points
        if show_stft and cfg.get('fd_stft') is not None:
            st = cfg['fd_stft']
            sb = st['f'] <= f_max
            sf = st['f'][sb]
            sm = np.abs(st['H_mid'][sb]) / norm
            ax_a.plot(sf, sm, marker=marker, ls='none', mfc='none',
                      mec=col, ms=7, mew=1.3, zorder=4)
            se = np.abs(st['H_eta'][sb]) * 100.0
            ax_c.plot(sf, se, marker=marker, ls='none', mfc='none',
                      mec=col, ms=7, mew=1.3, zorder=4)
            pm = _align_point_phase(sf, np.rad2deg(np.angle(st['H_mid'][sb])),
                                    fb, np.rad2deg(np.unwrap(np.angle(tf['H_mid'][band]))))
            ax_b.plot(sf, pm, marker=marker, ls='none', mfc='none', mec=col,
                      ms=7, mew=1.3, zorder=4)
            pe = _align_point_phase(sf, np.rad2deg(np.angle(st['H_eta'][sb])),
                                    fb, np.rad2deg(np.unwrap(np.angle(tf['H_eta'][band]))))
            ax_d.plot(sf, pe, marker=marker, ls='none', mfc='none', mec=col,
                      ms=7, mew=1.3, zorder=4)

    # Reference lines
    ax_c.axhline(100.0, color='k', lw=0.8, ls='-', alpha=0.4, zorder=0)
    ax_b.axhline(0.0, color='k', lw=0.8, alpha=0.3, zorder=0)
    ax_d.axhline(0.0, color='k', lw=0.8, alpha=0.3, zorder=0)

    ax_a.set_ylabel('normalized midpoint\namplitude [-]')
    ax_a.set_title('(a) midpoint amplitude')
    ax_b.set_ylabel('midpoint phase [deg]')
    ax_b.set_title('(b) midpoint phase')
    ax_c.set_ylabel('strain transfer\namplitude [%]')
    ax_c.set_title('(c) strain transfer amplitude')
    ax_d.set_ylabel('strain transfer phase [deg]')
    ax_d.set_title('(d) strain transfer phase')
    for ax in (ax_c, ax_d):
        ax.set_xlabel('frequency [Hz]')
    ax_a.set_xlim(0, f_max)
    ax_a.set_ylim(0, 1.05)

    handles, labels = ax_a.get_legend_handles_labels()
    if show_stft:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], marker='o', ls='none', mfc='none',
                              mec='k', ms=7, label='STFT cross-check'))
        labels.append('STFT cross-check')
    ax_a.legend(handles, labels, fontsize=8, loc='best')
    if single and show_eta_pred:
        ax_c.legend(fontsize=8, loc='best')

    fig.suptitle(title or 'Dynamic strain transfer — measured FRF '
                 '(cf. Fig. 3, Simone draft)', fontsize=13)
    plt.tight_layout()
    plt.show()
    return fig
