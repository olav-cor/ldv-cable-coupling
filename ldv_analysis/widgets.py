"""ipywidgets-based interactive browsers."""

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as ipw
from IPython.display import display

from .config import (N_CYCLES_PER_WINDOW, BANDWIDTH_FRAC, TARGET_FREQS, dataset_style,
                     LIN_FREQUENCIES, sweep_time_of_frequency)
from .signal import amplitude_spectrum, compute_spectrogram, find_dominant_peaks, bandpass, integrate_fft, compute_fk_spectrum
from .geometry import project_onto_chord
from .strain import strain_spatial_gradient, strain_3d_arclength
from .analysis import per_frequency_qc, linearity_qc
from .plotting import (plot_per_frequency_qc, _COMPONENT_LABELS, _select_1d_trace,
                       plot_spacetime)


def qc_dashboard(datasets):
    """Interactive QC: pick cable/frequency/window/bandwidth/gradient-direction
    and re-run per_frequency_qc on demand.
    """
    sty = {"description_width": "initial"}
    lay = ipw.Layout(width="400px")

    cable_w = ipw.Dropdown(options=[cfg['label'] for cfg in datasets],
                           description='Cable:', style=sty)
    freq_w = ipw.Dropdown(options=list(TARGET_FREQS) + [None], value=20,
                          description='Frequency [Hz]:', style=sty)
    custom_f = ipw.FloatText(value=20.0, description='Custom f [Hz]:',
                              style=sty, layout=lay)
    ncyc_w = ipw.IntSlider(value=N_CYCLES_PER_WINDOW, min=2, max=50, step=1,
                            description='Half-window [cycles]:',
                            style=sty, layout=lay)
    bw_w = ipw.FloatSlider(value=BANDWIDTH_FRAC, min=0.05, max=0.9, step=0.05,
                            description='Bandpass ± frac:', style=sty, layout=lay)
    dir_w = ipw.ToggleButtons(options=['chord', 'cartesian'], value='chord',
                                description='∂u/∂s direction:', style=sty)
    ref_w = ipw.ToggleButtons(options=['shaker', 'ends', 'shaker_end'], value='ends',
                                description='Reference:', style=sty)
    out = ipw.Output()

    def update(_=None):
        with out:
            out.clear_output(wait=True)
            cfg = next(c for c in datasets if c['label'] == cable_w.value)
            f_use = freq_w.value if freq_w.value is not None else custom_f.value
            res = per_frequency_qc(cfg, f_use,
                                    n_cycles=ncyc_w.value,
                                    bw_frac=bw_w.value,
                                    grad_direction=dir_w.value)
            if res is None:
                print("No valid window for this configuration.")
                return
            plot_per_frequency_qc(cfg, res, ref=ref_w.value)
            print(f"\nMedian eta (shaker): {res['eta_med']:.3f}   "
                  f"Median eta (ends): {res['eta_med_ends']:.3f}   "
                  f"Median eta (shaker_end): {res['eta_med_shend']:.3f}   "
                  f"(predicted eta = {cfg['eta_pred']:.3f})")

    for w in [cable_w, freq_w, custom_f, ncyc_w, bw_w, dir_w, ref_w]:
        w.observe(update, names='value')

    display(ipw.VBox([
        ipw.HBox([cable_w, freq_w]),
        custom_f,
        ipw.HBox([ncyc_w, bw_w]),
        ipw.HBox([dir_w, ref_w]),
        out,
    ]))
    update()


def dataset_sensor_browser(datasets, components=('vx', 'vy', 'vz'),
                           f_max=500.0, n_peaks=4):
    """Interactive trace/spectrum viewer: pick dataset → component → sensor.

    In spectrum view the top n_peaks dominant frequencies (0–f_max Hz) are
    marked with vertical dashed lines and listed in an annotation box.

    datasets : list of loaded cfg dicts.
    f_max    : upper frequency limit for peak search [Hz].
    n_peaks  : number of dominant peaks to annotate.
    """
    if not datasets:
        print("No datasets to browse.")
        return

    sty = {"description_width": "initial"}

    label_to_cfg = {cfg['label']: cfg for cfg in datasets}

    dataset_w = ipw.Dropdown(options=[cfg['label'] for cfg in datasets],
                              description='Dataset:', style=sty)
    component_w = ipw.Dropdown(options=list(components), value=components[0],
                                description='Component:', style=sty)
    initial_ns = datasets[0]['vx'].shape[1]
    sensor_w = ipw.IntSlider(value=0, min=0, max=initial_ns - 1, step=1,
                              description='Sensor index:', style=sty,
                              continuous_update=False)
    view_w = ipw.ToggleButtons(options=['time', 'spectrum', 'fk'], value='time',
                                description='View:', style=sty)
    logy_w = ipw.Checkbox(value=True, description='log-y (spectrum)', style=sty)
    fmax_w = ipw.FloatSlider(value=f_max, min=50.0, max=2500.0, step=50.0,
                              description='f_max [Hz]:', style=sty,
                              continuous_update=False,
                              layout=ipw.Layout(width='350px'))
    fk_db_w = ipw.IntSlider(value=-40, min=-80, max=-10, step=5,
                             description='FK dB floor:', style=sty,
                             continuous_update=False,
                             layout=ipw.Layout(width='280px'))
    out = ipw.Output()

    def _sync_sensor_range(*_):
        cfg = label_to_cfg[dataset_w.value]
        n_s = cfg[component_w.value].shape[1] if cfg[component_w.value].ndim == 2 else 1
        sensor_w.max = max(0, n_s - 1)
        if sensor_w.value > sensor_w.max:
            sensor_w.value = sensor_w.max

    def update(_=None):
        _sync_sensor_range()
        cfg = label_to_cfg[dataset_w.value]
        comp = component_w.value
        s_idx = sensor_w.value
        col, _m, _fs = dataset_style(cfg)
        f_lim = fmax_w.value

        arr = cfg[comp]
        view = view_w.value

        with out:
            out.clear_output(wait=True)

            if view == 'fk':
                if arr.ndim != 2:
                    print(f"F-K spectrum requires 2-D data (N_t × N_sensors). "
                          f"'{comp}' is 1-D.")
                    return
                freqs, k, FK_db, dx = compute_fk_spectrum(arr, cfg['fs'], cfg['x'])
                f_mask = freqs <= f_lim
                fig, ax = plt.subplots(figsize=(9, 4.5))
                pcm = ax.pcolormesh(k, freqs[f_mask], FK_db[f_mask, :],
                                    cmap='inferno', vmin=fk_db_w.value, vmax=0.0,
                                    shading='auto')
                plt.colorbar(pcm, ax=ax, label='dB (vs peak)')
                ax.axvline(0, color='white', lw=0.6, ls='--', alpha=0.4)
                ax.set_xlabel('Wavenumber [1/m]')
                ax.set_ylabel('Frequency [Hz]')
                ax.set_title(f"{cfg['label']} — {comp} — F-K spectrum  "
                             f"(dx ≈ {dx*1e3:.1f} mm, all sensors)")
                plt.tight_layout()
                plt.show()
                return

            sig = arr if arr.ndim == 1 else arr[:, s_idx]
            fig, ax = plt.subplots(figsize=(11, 3.5))
            if view == 'time':
                ax.plot(cfg['t'], sig * 1e3, lw=0.5, color=col)
                ax.set_xlabel('Time [s]'); ax.set_ylabel('Amplitude [mm/s]')
                ax.set_title(f"{cfg['label']} — {comp} — sensor {s_idx}")
            else:  # spectrum
                freqs, spec = amplitude_spectrum(sig[:, np.newaxis], cfg['fs'])
                spec_mm = spec.ravel() * 1e3
                ax.plot(freqs, spec_mm, lw=0.8, color=col)
                ax.set_xscale('log')
                if logy_w.value:
                    ax.set_yscale('log')
                ax.set_xlim(right=f_lim)
                ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('|V| [mm/s]')
                ax.set_title(f"{cfg['label']} — {comp} — sensor {s_idx} (spectrum)")

                try:
                    peaks = find_dominant_peaks(freqs, spec.ravel(),
                                                n_peaks=n_peaks, f_max=f_lim)
                    s_peak = spec.ravel()[freqs <= f_lim].max() + 1e-30
                    box_lines = []
                    for rank, (fp, ap) in enumerate(peaks, 1):
                        amp_mm = ap * s_peak * 1e3
                        ax.axvline(fp, color=col, lw=0.9, ls='--', alpha=0.65)
                        ax.plot(fp, amp_mm, 'v', color=col, ms=8, zorder=5)
                        box_lines.append(f'#{rank}  {fp:6.1f} Hz   {ap:.3f}')
                    ax.text(0.97, 0.97, '\n'.join(box_lines),
                            transform=ax.transAxes, fontsize=8,
                            va='top', ha='right', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='white',
                                      alpha=0.88, edgecolor='gray', lw=0.8))
                except Exception:
                    pass

            plt.tight_layout()
            plt.show()

    for w in [dataset_w, component_w, sensor_w, view_w, logy_w, fmax_w, fk_db_w]:
        w.observe(update, names='value')

    display(ipw.VBox([
        ipw.HBox([dataset_w, component_w]),
        sensor_w,
        ipw.HBox([view_w, logy_w, fk_db_w]),
        fmax_w,
        out,
    ]))
    update()


def spectrogram_browser(sources, components=('vx', 'vy', 'vz')):
    """Interactive spectrogram viewer.

    `sources` is a list of dicts: either loaded cable cfgs (2-D components
    with N_sensors) or shaker-reference dicts (1-D components). Each source
    is identified by its 'label' (cfg) or 'path' (shaker-ref). The sensor
    slider auto-disables for 1-D sources.

    Controls: source dropdown, sensor slider, f_max, nperseg, db_floor.
    Shows all selected components stacked as a spectrogram grid.
    """
    if not sources:
        print("No sources to browse.")
        return

    sty = {"description_width": "initial"}

    def _name(src):
        return src.get('label', src.get('path', '<unnamed>'))

    name_to_src = {_name(s): s for s in sources}

    src_w = ipw.Dropdown(options=list(name_to_src.keys()),
                          description='Source:', style=sty)

    def _max_sensor(src):
        arr = np.asarray(src[components[0]])
        return arr.shape[1] - 1 if arr.ndim == 2 else 0

    sensor_w = ipw.IntSlider(value=0, min=0, max=_max_sensor(sources[0]),
                              step=1, description='Sensor index:',
                              style=sty, continuous_update=False)
    fmax_w = ipw.FloatSlider(value=600.0, min=50.0, max=2500.0, step=50.0,
                              description='f_max [Hz]:', style=sty,
                              continuous_update=False)
    nperseg_w = ipw.IntSlider(value=int(sources[0]['fs'] // 2),
                               min=64, max=int(sources[0]['fs'] * 2), step=32,
                               description='nperseg:', style=sty,
                               continuous_update=False)
    db_w = ipw.IntSlider(value=-80, min=-120, max=-20, step=5,
                          description='dB floor:', style=sty,
                          continuous_update=False)
    out = ipw.Output()

    def _sync_sensor_range(*_):
        src = name_to_src[src_w.value]
        n_max = _max_sensor(src)
        sensor_w.max = n_max
        sensor_w.disabled = (n_max == 0)
        if sensor_w.value > n_max:
            sensor_w.value = n_max
        # Also re-cap nperseg if a smaller signal is selected
        n_samples = np.asarray(src[components[0]]).shape[0]
        nperseg_w.max = max(nperseg_w.min, n_samples)
        if nperseg_w.value > nperseg_w.max:
            nperseg_w.value = nperseg_w.max

    def update(_=None):
        _sync_sensor_range()
        src = name_to_src[src_w.value]
        fs = src['fs']
        s_idx = sensor_w.value if not sensor_w.disabled else None

        with out:
            out.clear_output(wait=True)
            n = len(components)
            fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n),
                                     sharex=True, squeeze=False)
            axes = axes[:, 0]
            for i, comp in enumerate(components):
                try:
                    x = _select_1d_trace(src, comp, sensor_index=s_idx)
                except ValueError as e:
                    axes[i].text(0.5, 0.5, str(e), ha='center', va='center',
                                  transform=axes[i].transAxes)
                    continue
                f, tt, Sxx = compute_spectrogram(x, fs, nperseg=nperseg_w.value)
                P = Sxx ** 2
                Sdb = 10.0 * np.log10(P / (P.max() + 1e-30) + 1e-30)
                pcm = axes[i].pcolormesh(tt, f, Sdb, cmap='viridis',
                                          vmin=db_w.value, vmax=0.0, shading='auto')
                axes[i].set_ylabel(f"{_COMPONENT_LABELS.get(comp, comp)}\nFrequency [Hz]")
                axes[i].set_ylim(top=fmax_w.value)
                plt.colorbar(pcm, ax=axes[i], label='dB (vs peak)', shrink=0.85)
            axes[-1].set_xlabel('Time [s]')
            suffix = f"  (sensor {s_idx})" if s_idx is not None else ""
            fig.suptitle(f"Spectrogram — {src_w.value}{suffix}", fontsize=12, y=1.0)
            plt.tight_layout()
            plt.show()

    for w in [src_w, sensor_w, fmax_w, nperseg_w, db_w]:
        w.observe(update, names='value')

    display(ipw.VBox([
        src_w,
        ipw.HBox([sensor_w, fmax_w]),
        ipw.HBox([nperseg_w, db_w]),
        out,
    ]))
    update()


def linearity_dashboard(datasets, frequencies=None):
    """Interactive QC for linearity-test data.

    Shows per-frequency coupling efficiency results with three view modes:
      * η(t) — time trace of η with amplitude envelope overlay.
      * η vs amplitude — scatter of η against |δL| (the key linearity check).
      * raw waveforms — bandpassed vx/vy/vz space-time wavefields.

    Requires cfg['lin_results'] to be populated (call run_linearity_analysis first).
    Geometry (chord_unit, eta_pred) must also be set (call prepare_geometry first).

    Parameters
    ----------
    datasets    : list of loaded linearity cfg dicts.
    frequencies : subset of LIN_FREQUENCIES to show; default = LIN_FREQUENCIES.
    """
    from scipy.ndimage import median_filter

    if not datasets:
        print("No datasets.")
        return

    sty = {"description_width": "initial"}
    if frequencies is None:
        frequencies = LIN_FREQUENCIES

    label_to_cfg = {cfg['label']: cfg for cfg in datasets}

    cable_w = ipw.Dropdown(options=[cfg['label'] for cfg in datasets],
                            description='Cable:', style=sty)
    freq_w = ipw.Dropdown(options=[str(f) for f in frequencies],
                           value=str(frequencies[0]),
                           description='Frequency [Hz]:', style=sty)
    view_w = ipw.ToggleButtons(
        options=['η(t)', 'η vs amplitude', 'raw waveforms'],
        value='η(t)', description='View:', style=sty)
    ref_w = ipw.ToggleButtons(options=['shaker', 'ends', 'shaker_end'], value='shaker',
                               description='Reference:', style=sty)
    smooth_w = ipw.IntSlider(value=5, min=1, max=20, step=1,
                              description='Smooth [cycles]:', style=sty,
                              layout=ipw.Layout(width='380px'),
                              continuous_update=False)
    fade_w = ipw.Checkbox(value=True, description='Ramp only (exclude fade)',
                           style=sty)
    nbins_w = ipw.IntSlider(value=25, min=5, max=60, step=5,
                             description='Amp. bins:', style=sty,
                             layout=ipw.Layout(width='300px'),
                             continuous_update=False)
    out = ipw.Output()

    def update(_=None):
        cfg = label_to_cfg[cable_w.value]
        f_target = float(freq_w.value)
        results = cfg.get('lin_results', {})
        if ref_w.value == 'ends':
            eta_key, eta_med_key = 'eta_t_ends', 'eta_med_ends'
        elif ref_w.value == 'shaker_end':
            eta_key, eta_med_key = 'eta_t_shend', 'eta_med_shend'
        else:
            eta_key, eta_med_key = 'eta_t', 'eta_med'
        col, _, _fs = dataset_style(cfg) if 'gap_m' in cfg else ('#555', 'o', 'full')
        eta_pred = cfg.get('eta_pred', 1.0)

        with out:
            out.clear_output(wait=True)

            if f_target not in results:
                print(f"No result for {f_target} Hz. Run run_linearity_analysis first.")
                return

            res = results[f_target]
            t_rel = res['t_win'] - res['t_start']
            amp_um = res['amp_env'] * 1e6
            eta_raw = res[eta_key].copy()
            t_ramp_end = res['t_ramp_end']

            win = max(int(smooth_w.value / f_target * cfg['fs']), 5)
            eta_fill = np.where(np.isnan(eta_raw), 0.0, eta_raw)
            eta_smooth = median_filter(eta_fill, size=win).astype(float)
            eta_smooth[np.isnan(eta_raw)] = np.nan

            view = view_w.value

            if view == 'η(t)':
                fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
                ax_top, ax_bot = axes

                ax_top.axvspan(t_ramp_end, t_rel[-1] + 0.01,
                               alpha=0.13, color='gray', label='fade')
                ax_top.axhline(1.0, color='gray', ls='--', lw=0.8)
                ax_top.axhline(eta_pred, color='k', ls=':', lw=1.0,
                               label=f'η_pred={eta_pred:.3f}')
                ax_top.plot(t_rel, eta_raw, color=col, lw=0.3, alpha=0.35)
                ax_top.plot(t_rel, eta_smooth, color=col, lw=1.6,
                            label=f'η_med={res[eta_med_key]:.3f}')
                ax_top.set_ylim(-0.3, 2.3)
                ax_top.set_ylabel('η = δxₗ / δL')
                ax_top.legend(fontsize=9, loc='upper right')
                ax_top.set_title(f'{cfg["label"]} — {f_target:.0f} Hz  '
                                 f'[{ref_w.value}]')

                ax_bot.axvspan(t_ramp_end, t_rel[-1] + 0.01,
                               alpha=0.13, color='gray')
                ax_bot.plot(t_rel, amp_um, color='steelblue', lw=1.0)
                ax_bot.set_ylabel('|δL| [μm]')
                ax_bot.set_xlabel('Time in segment [s]')
                ax_bot.set_title(f'Amplitude envelope — peak = {amp_um.max():.2f} μm')

            elif view == 'η vs amplitude':
                if fade_w.value:
                    sel = res['ramp_mask']
                else:
                    sel = np.ones(len(t_rel), dtype=bool)

                amp_s, eta_s = amp_um[sel], eta_smooth[sel]
                valid = ~np.isnan(eta_s) & (amp_s > 0)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                ax_sc, ax_bin = axes

                if valid.sum() > 5:
                    t_sel = t_rel[sel]
                    sc = ax_sc.scatter(amp_s[valid], eta_s[valid], s=4,
                                       c=t_sel[valid], cmap='plasma', alpha=0.5)
                    plt.colorbar(sc, ax=ax_sc, label='Time in segment [s]')

                    amp_v, eta_v = amp_s[valid], eta_s[valid]
                    n_bins = nbins_w.value
                    edges = np.linspace(amp_v.min(), amp_v.max(), n_bins + 1)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    bin_med = [np.nanmedian(eta_v[(amp_v >= lo) & (amp_v < hi)])
                               if ((amp_v >= lo) & (amp_v < hi)).sum() > 3 else np.nan
                               for lo, hi in zip(edges[:-1], edges[1:])]
                    ax_bin.plot(centers, bin_med, 'o-', color=col, ms=6, lw=1.8)

                for ax in axes:
                    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
                    ax.axhline(eta_pred, color='k', ls=':', lw=0.9,
                               label=f'η_pred={eta_pred:.3f}')
                    ax.set_xlabel('|δL| [μm]')
                    ax.set_ylabel('η')
                    ax.set_ylim(-0.2, 2.2)
                    ax.legend(fontsize=8)

                ax_sc.set_title('Scatter (colour = time in segment)')
                ax_bin.set_title(f'Binned median η  ({nbins_w.value} bins)')
                fade_lbl = ' — ramp only' if fade_w.value else ''
                fig.suptitle(f'{cfg["label"]} — {f_target:.0f} Hz{fade_lbl}',
                             fontsize=11)

            else:  # raw waveforms
                x = cfg['x']
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                for ax, arr, lbl in zip(axes,
                                        [res['vx'], res['vy'], res['vz']],
                                        ['$v_x$', '$v_y$', '$v_z$']):
                    plot_spacetime(ax, x, t_rel, arr, unit='[m/s]')
                    ax.set_title(f'{lbl}  [{f_target:.0f} Hz, bandpassed]')
                    ax.set_ylabel('Time in segment [s]')
                fig.suptitle(f'{cfg["label"]} — raw bandpassed wavefields',
                             fontsize=11)

            plt.tight_layout()
            plt.show()

    for w in [cable_w, freq_w, view_w, ref_w, smooth_w, fade_w, nbins_w]:
        w.observe(update, names='value')

    display(ipw.VBox([
        ipw.HBox([cable_w, freq_w]),
        view_w,
        ipw.HBox([ref_w, fade_w]),
        ipw.HBox([smooth_w, nbins_w]),
        out,
    ]))
    update()


def strain_comparison_dashboard(datasets):
    """Interactive comparison of Method 1 (spatial gradient) vs Method 2 (arc-length).

    Three stacked panels with a shared time axis:
      ① Reference elongation   — shaker δL  or  end-sensor Δu  [mm]
      ② Method 1: ∫ε ds        — spatial-gradient strain, trapezoid-integrated [mm]
      ③ Method 2: Σδd_i        — arc-length per-segment elongation summed  [mm]

    The "N sensors" radio lets you subsample the sensor array to 9 / 5 / 3
    evenly-spaced points so you can see how spatial resolution affects each method.

    Requires cfg['chord_unit'] to be set (call prepare_geometry first).
    """
    sty = {"description_width": "initial"}
    lay_w = ipw.Layout(width="420px")

    label_to_cfg = {cfg['label']: cfg for cfg in datasets}

    cable_w = ipw.Dropdown(options=[cfg['label'] for cfg in datasets],
                           description='Cable:', style=sty)
    freq_w = ipw.Dropdown(options=list(TARGET_FREQS) + [None], value=20,
                          description='Frequency [Hz]:', style=sty)
    custom_f = ipw.FloatText(value=20.0, description='Custom f [Hz]:',
                              style=sty, layout=lay_w)
    ncyc_w = ipw.IntSlider(value=N_CYCLES_PER_WINDOW, min=2, max=50, step=1,
                            description='Half-window [cycles]:', style=sty,
                            layout=lay_w, continuous_update=False)
    bw_w = ipw.FloatSlider(value=BANDWIDTH_FRAC, min=0.05, max=0.9, step=0.05,
                            description='Bandpass ± frac:', style=sty,
                            layout=lay_w, continuous_update=False)
    ref_w = ipw.ToggleButtons(options=['shaker', 'ends', 'shaker_end'], value='shaker',
                               description='Reference:', style=sty)
    dir_w = ipw.ToggleButtons(options=['chord', 'cartesian'], value='chord',
                               description='∂u/∂s direction:', style=sty)
    npts_w = ipw.RadioButtons(options=['all', '9', '5', '3'], value='all',
                               description='N sensors:', style=sty)
    out = ipw.Output()

    def update(_=None):
        cfg = label_to_cfg[cable_w.value]
        if 'chord_unit' not in cfg:
            with out:
                out.clear_output(wait=True)
                print("Geometry not prepared. Call prepare_geometry(cfg) first.")
            return

        f_use = freq_w.value if freq_w.value is not None else custom_f.value
        fs, dt, t = cfg['fs'], cfg['dt'], cfg['t']
        chord_unit = cfg['chord_unit']

        # Time window
        t_c = sweep_time_of_frequency(f_use)
        half = ncyc_w.value / f_use
        mask = (t >= max(t[0], t_c - half)) & (t <= min(t[-1], t_c + half))
        t_win = t[mask]

        with out:
            out.clear_output(wait=True)
            if len(t_win) < 32:
                print("Window too short for this frequency / n_cycles combination.")
                return

            bw = bw_w.value
            f_lo_bp, f_hi_bp = f_use * (1 - bw), f_use * (1 + bw)

            # Bandpass + integrate cable signals
            vx_bp = bandpass(cfg['vx'], fs, f_lo_bp, f_hi_bp)[mask]
            vy_bp = bandpass(cfg['vy'], fs, f_lo_bp, f_hi_bp)[mask]
            vz_bp = bandpass(cfg['vz'], fs, f_lo_bp, f_hi_bp)[mask]
            ux = integrate_fft(vx_bp, dt)
            uy = integrate_fft(vy_bp, dt)
            uz = integrate_fft(vz_bp, dt)

            # Reference trace
            idx_l, idx_r = cfg['idx_left'], cfg['idx_right']
            u_axial_ends = project_onto_chord(ux[:, [idx_l, idx_r]],
                                              uy[:, [idx_l, idx_r]],
                                              uz[:, [idx_l, idx_r]], chord_unit)
            if ref_w.value in ('shaker', 'shaker_end'):
                sh_ux = integrate_fft(bandpass(cfg['sh_vx'], fs, f_lo_bp, f_hi_bp)[mask], dt)
                sh_uy = integrate_fft(bandpass(cfg['sh_vy'], fs, f_lo_bp, f_hi_bp)[mask], dt)
                sh_uz = integrate_fft(bandpass(cfg['sh_vz'], fs, f_lo_bp, f_hi_bp)[mask], dt)
                delta_L = project_onto_chord(sh_ux, sh_uy, sh_uz, chord_unit)
                if cfg.get('shaker_end') == 'right':
                    delta_L = -delta_L

            if ref_w.value == 'shaker':
                ref_trace = delta_L
                ref_label = 'Shaker  δL  (chord-projected)'
            elif ref_w.value == 'shaker_end':
                if cfg.get('shaker_end') == 'right':
                    ref_trace = delta_L + u_axial_ends[:, 1]
                else:
                    ref_trace = delta_L - u_axial_ends[:, 0]
                ref_label = 'Shaker − end-sensor  δL_shend'
            else:
                ref_trace = u_axial_ends[:, 1] - u_axial_ends[:, 0]
                ref_label = 'End-sensor  Δu  (chord-projected)'

            # Sensor subsampling
            idx_l, idx_r = cfg['idx_left'], cfg['idx_right']
            n_avail = idx_r - idx_l + 1
            npts_str = npts_w.value
            n_req = n_avail if npts_str == 'all' else min(int(npts_str), n_avail)
            sub_idx = np.round(np.linspace(idx_l, idx_r, n_req)).astype(int)
            n_used = len(sub_idx)

            XYZ_sub = cfg['XYZ'][sub_idx]
            ux_sub, uy_sub, uz_sub = ux[:, sub_idx], uy[:, sub_idx], uz[:, sub_idx]

            # Method 1: spatial-gradient strain → trapezoid-integrate over chord span
            direction = dir_w.value
            sg = strain_spatial_gradient(XYZ_sub, ux_sub, uy_sub, uz_sub,
                                         chord_unit, direction=direction)
            s_sub = np.array([np.dot(XYZ_sub[i] - XYZ_sub[0], chord_unit)
                              for i in range(n_used)])
            delta_xl_m1 = np.trapz(np.where(np.isnan(sg), 0.0, sg), s_sub, axis=1)

            # Method 2: arc-length per-segment summed
            _, _, _, delta_xl_m2 = strain_3d_arclength(
                XYZ_sub, ux_sub, uy_sub, uz_sub, 0, n_used - 1)

            col, _, _ = dataset_style(cfg)
            t_rel = t_win - t_win[0]
            sc = 1e3  # m → mm

            def _pp(arr):
                return (np.nanmax(arr) - np.nanmin(arr)) * sc

            def _xcorr_norm(a, b):
                """Normalised cross-correlation, returns (lags_s, cc)."""
                a = a - a.mean()
                b = b - b.mean()
                cc = np.correlate(a, b, mode='full')
                norm = np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-30
                cc /= norm
                lags = (np.arange(len(cc)) - (len(a) - 1)) / fs
                return lags, cc

            fig, (ax_tr, ax_cc) = plt.subplots(
                2, 1, figsize=(13, 7.5),
                gridspec_kw={'height_ratios': [3, 2]})

            # ── Top panel: all three traces overlaid ──────────────────────────
            ax_tr.plot(t_rel, ref_trace * sc,  color='steelblue',  lw=1.2, alpha=0.9,
                       label=f'① Ref ({ref_label.split()[0]})  pp={_pp(ref_trace):.4f} mm')
            ax_tr.plot(t_rel, delta_xl_m1 * sc, color='darkorange', lw=1.2, alpha=0.85,
                       label=f'② M1 ∫ε ds ({direction})  pp={_pp(delta_xl_m1):.4f} mm')
            ax_tr.plot(t_rel, delta_xl_m2 * sc, color='seagreen',  lw=1.2, alpha=0.85,
                       label=f'③ M2 Σδd_i            pp={_pp(delta_xl_m2):.4f} mm')
            ax_tr.axhline(0, color='gray', lw=0.5, ls='--')
            ax_tr.set_ylabel('Elongation [mm]')
            ax_tr.legend(fontsize=9, loc='upper right',
                         framealpha=0.9, handlelength=2.5)
            ax_tr.set_title(
                f"{cfg['label']}  —  {f_use:.0f} Hz  "
                f"(±{bw*100:.0f}% BP,  {ncyc_w.value} half-cycles,  "
                f"{n_used} sensors,  ref: {ref_w.value})",
                fontsize=11)

            # ── Bottom panel: normalised cross-correlations ───────────────────
            lags1, cc1 = _xcorr_norm(delta_xl_m1, ref_trace)
            lags2, cc2 = _xcorr_norm(delta_xl_m2, ref_trace)
            peak1 = cc1[np.argmax(np.abs(cc1))]
            peak2 = cc2[np.argmax(np.abs(cc2))]
            lag1_ms = lags1[np.argmax(np.abs(cc1))] * 1e3
            lag2_ms = lags2[np.argmax(np.abs(cc2))] * 1e3

            ax_cc.plot(lags1 * 1e3, cc1, color='darkorange', lw=1.0, alpha=0.9,
                       label=f'M1 vs Ref   peak={peak1:.3f}  @{lag1_ms:+.1f} ms')
            ax_cc.plot(lags2 * 1e3, cc2, color='seagreen',  lw=1.0, alpha=0.9,
                       label=f'M2 vs Ref   peak={peak2:.3f}  @{lag2_ms:+.1f} ms')
            ax_cc.axvline(0, color='gray', lw=0.8, ls='--')
            ax_cc.axhline(0, color='gray', lw=0.4, ls=':')

            # Zoom to ±3 cycles so the main peak is always visible
            lag_zoom = 3e3 / f_use
            ax_cc.set_xlim(-lag_zoom, lag_zoom)

            ax_cc.set_xlabel('Lag [ms]')
            ax_cc.set_ylabel('Normalised\ncross-correlation')
            ax_cc.set_title('Cross-correlation with reference (normalised)')
            ax_cc.legend(fontsize=9, loc='upper right', framealpha=0.9)

            plt.tight_layout()
            plt.show()

    for w in [cable_w, freq_w, custom_f, ncyc_w, bw_w, ref_w, dir_w, npts_w]:
        w.observe(update, names='value')

    display(ipw.VBox([
        ipw.HBox([cable_w, freq_w]),
        custom_f,
        ipw.HBox([ncyc_w, bw_w]),
        ipw.HBox([ref_w, dir_w]),
        npts_w,
        out,
    ]))
    update()
