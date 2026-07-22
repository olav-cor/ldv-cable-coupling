"""Publication-figure infrastructure for the thesis.

Design rules (differ from the notebook QC figures):
* **No titles on the figure** — the description belongs in the LaTeX caption.
  Use :func:`caption_note` to print the would-be title as text instead.
* All sizes/limits/labels controlled from one place: pass a ``spec`` dict to
  :func:`paper_figure` / :func:`style_axis`, or edit :data:`STYLE` once at the
  top of the figure notebook.
* Figures are saved via :func:`save_paper_fig` to ``Figures_paper/`` as PNG
  (for quick inspection) and PDF (vector, for LaTeX).

Typical cell in 07_Paper_Figures.ipynb::

    from ldv_analysis import paper
    paper.apply_paper_style()

    fig, ax = paper.paper_figure(width='half')
    ax.plot(f, eta * 100, color='k')
    paper.style_axis(ax, xlabel='frequency [Hz]',
                     ylabel='strain transfer efficiency [%]',
                     xlim=(0, 60), ylim=(0, 120))
    paper.save_paper_fig(fig, 'eta_frf_C1')
    paper.caption_note('Strain-transfer FRF of cable C1, 10 cm gap.')
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from .config import FIG_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Global style — edit in one place (or override per call)
# ─────────────────────────────────────────────────────────────────────────────
# A4 text width with 2.5 cm margins ≈ 16 cm ≈ 6.3 in.
TEXT_WIDTH_IN = 6.3

STYLE = {
    # canvas
    'width_full': TEXT_WIDTH_IN,          # full text width [in]
    'width_half': 0.62 * TEXT_WIDTH_IN,   # single-column-ish [in]
    'aspect': 0.62,                       # default height = aspect * width
    'dpi': 300,

    # typography
    'font_family': 'serif',
    'font_serif': ['STIX Two Text', 'Libertinus Serif', 'DejaVu Serif'],
    'mathtext': 'stix',
    'label_fs': 10,
    'tick_fs': 9,
    'legend_fs': 8,
    'panel_fs': 11,

    # lines / markers
    'line_lw': 1.4,
    'marker_size': 6.5,
    'edge_lw': 0.6,

    # configuration markers (see config_marker_kw) — the style of the
    # "eta vs Theta across all configurations" figure: filled marker in the
    # cable colour with a black edge and some transparency; intended-sag
    # variants are drawn open in the cable colour.
    'marker_alpha': 0.75,
    'marker_edge_color': 'black',
    'marker_edge_lw': 0.7,
    'marker_sag_edge_lw': 1.4,

    # axes cosmetic
    'grid': True,
    'spine_lw': 0.8,

    # output
    'formats': ('png' , 'pdf'),
}


def apply_paper_style(overrides=None):
    """Set matplotlib rcParams for paper figures. Call once per notebook.

    overrides: dict merged into STYLE before applying (e.g. {'label_fs': 11}).
    """
    if overrides:
        STYLE.update(overrides)
    plt.rcParams.update({
        'font.family': STYLE['font_family'],
        'font.serif': STYLE['font_serif'],
        'mathtext.fontset': STYLE['mathtext'],
        'font.size': STYLE['tick_fs'],
        'axes.labelsize': STYLE['label_fs'],
        'axes.titlesize': STYLE['label_fs'],   # titles unused, kept consistent
        'legend.fontsize': STYLE['legend_fs'],
        'xtick.labelsize': STYLE['tick_fs'],
        'ytick.labelsize': STYLE['tick_fs'],
        'axes.linewidth': STYLE['spine_lw'],
        'lines.linewidth': STYLE['line_lw'],
        'axes.grid': STYLE['grid'],
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 110,                     # screen preview
        'savefig.dpi': STYLE['dpi'],
        'figure.autolayout': False,
    })


def paper_figure(nrows=1, ncols=1, width='full', height=None, **subplots_kw):
    """Create a paper-sized figure.

    width  : 'full' (text width), 'half', or a float in inches.
    height : float in inches; default = STYLE['aspect'] * width (per row).
    Extra kwargs go to plt.subplots (sharex, gridspec_kw, ...).
    """
    if width == 'full':
        w = STYLE['width_full']
    elif width == 'half':
        w = STYLE['width_half']
    else:
        w = float(width)
    h = height if height is not None else STYLE['aspect'] * w * nrows / max(ncols, 1)
    return plt.subplots(nrows, ncols, figsize=(w, h), **subplots_kw)


def style_axis(ax, xlabel=None, ylabel=None, xlim=None, ylim=None,
               xscale=None, yscale=None, legend=False, legend_kw=None,
               grid=None):
    """Apply the per-axis controls in one call (all optional).

    grid : None = follow STYLE['grid'] (on by default), or True/False to force
           it for this axis. Log axes also get the minor gridlines.

    Never sets a title — captions live in the LaTeX document.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    grid = STYLE['grid'] if grid is None else grid
    if grid:
        ax.grid(True, which='major', alpha=0.3, ls='--', lw=0.6, zorder=0)
        if ax.get_xscale() == 'log' or ax.get_yscale() == 'log':
            ax.grid(True, which='minor', alpha=0.15, ls=':', lw=0.4, zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)

    if legend:
        kw = dict(frameon=False)
        kw.update(legend_kw or {})
        ax.legend(**kw)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Configuration markers (colour = cable, shape = gap, open = intended sag)
# ─────────────────────────────────────────────────────────────────────────────

def config_marker_kw(cable=None, gap_m=None, is_sag=False, color=None,
                     marker=None, size=None, **overrides):
    """Line2D/plot kwargs for one measurement configuration.

    Filled marker in the cable colour with a black edge and STYLE['marker_alpha']
    transparency; intended-sag variants are drawn open in the cable colour.

    cable  : 'C1'…'C7' (looked up in CABLE_COLORS) — or pass `color` directly.
    gap_m  : nominal gap in metres (looked up in GAP_MARKERS) — or pass `marker`.
    is_sag : True for the intended-sag variants (open marker).
    Extra keyword arguments override the result.
    """
    from .config import CABLE_COLORS, GAP_MARKERS

    if color is None:
        color = CABLE_COLORS.get(str(cable), '#888888')
    if marker is None:
        marker = GAP_MARKERS.get(round(float(gap_m), 2), 's') if gap_m is not None else 'o'

    kw = dict(
        marker=marker,
        linestyle='none',
        markersize=STYLE['marker_size'] if size is None else size,
        alpha=STYLE['marker_alpha'],
        markerfacecolor='none' if is_sag else color,
        markeredgecolor=color if is_sag else STYLE['marker_edge_color'],
        markeredgewidth=(STYLE['marker_sag_edge_lw'] if is_sag
                         else STYLE['marker_edge_lw']),
        color=color,
    )
    kw.update(overrides)
    return kw


def config_point(ax, x, y, cable=None, gap_m=None, is_sag=False,
                 xerr=None, yerr=None, color=None, marker=None, size=None,
                 zorder=5, capsize=2.5, elinewidth=0.8, **overrides):
    """Draw one configuration point with optional x/y error bars.

    The marker and the error bars are drawn separately (as in the notebook-03
    eta-vs-Theta figure) so the bars keep full opacity and the cable colour
    while the marker itself stays semi-transparent.
    """
    kw = config_marker_kw(cable=cable, gap_m=gap_m, is_sag=is_sag,
                          color=color, marker=marker, size=size, **overrides)
    ecolor = kw['color']
    if xerr is not None or yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', ecolor=ecolor,
                    elinewidth=elinewidth, capsize=capsize,
                    capthick=elinewidth, alpha=0.7, zorder=zorder - 1)
    return ax.plot(x, y, zorder=zorder, **kw)


def config_handles(cables=None, gaps=None, sag=True, size=None):
    """Legend handles for the configuration encoding.

    Returns (cable_handles, gap_handles, sag_handles); pass any combination to
    ``ax.legend(handles=…)``. `cables`/`gaps` default to the full catalogue.
    """
    from matplotlib.lines import Line2D
    from .config import CABLE_COLORS, GAP_MARKERS

    size = STYLE['marker_size'] if size is None else size
    cables = list(CABLE_COLORS) if cables is None else list(cables)
    gaps = list(GAP_MARKERS) if gaps is None else list(gaps)

    cable_handles = [
        Line2D([0], [0], label=name,
               **config_marker_kw(cable=name, marker='o', size=size))
        for name in cables]
    gap_handles = [
        Line2D([0], [0], label=f'{int(round(g * 100))} cm',
               **config_marker_kw(color='#9a9a9a', gap_m=g, size=size))
        for g in gaps]
    sag_handles = []
    if sag:
        sag_handles = [
            Line2D([0], [0], label='taut',
                   **config_marker_kw(color='#555555', marker='o', size=size)),
            Line2D([0], [0], label='intended sag',
                   **config_marker_kw(color='#555555', marker='o',
                                      is_sag=True, size=size)),
        ]
    return cable_handles, gap_handles, sag_handles


def panel_label(ax, letter, x=0.02, y=0.97, boxed=False):
    """Bold panel letter ('a', 'b', …) in the axis corner."""
    bbox = (dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5)
            if boxed else None)
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=STYLE['panel_fs'], fontweight='bold',
            va='top', ha='left', bbox=bbox)


def add_headroom(ax, frac=0.08, skip_linestyles=(':',)):
    """Expand the y-limits so panel labels / legends don't overlap the data."""
    y = []
    for line in ax.get_lines():
        if line.get_linestyle() in skip_linestyles:
            continue
        y.extend(np.asarray(line.get_ydata(), dtype=float))
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    if y.size:
        lo, hi = y.min(), y.max()
        if hi > lo:
            ax.set_ylim(lo - 0.02 * (hi - lo), hi + frac * (hi - lo))


def save_paper_fig(fig, name, out_dir=None, formats=None, dpi=None,
                   tight=True):
    """Save the figure to Figures_paper/<name>.<fmt> for every format.

    Returns the list of written paths.
    """
    out_dir = FIG_DIR if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    formats = formats or STYLE['formats']
    dpi = dpi or STYLE['dpi']
    kw = dict(dpi=dpi)
    if tight:
        fig.tight_layout()
        kw['bbox_inches'] = 'tight'
    paths = []
    for fmt in formats:
        p = os.path.join(str(out_dir), f'{name}.{fmt}')
        fig.savefig(p, **kw)
        paths.append(p)
    print('saved: ' + ', '.join(os.path.basename(p) for p in paths))
    return paths


def caption_note(text):
    """Print the figure description as notebook text (instead of a title)."""
    print(f'[caption] {text}')


_GEOM_VIEWS = {'side': (2, 'z', 'side view (x–z)'),
               'top':  (1, 'y', 'top view (x–y)')}


def _chord_deviation(x, q):
    """Deviation of q(x) from the straight line through its two endpoints."""
    x, q = np.asarray(x, float), np.asarray(q, float)
    dx = x[-1] - x[0]
    frac = (x - x[0]) / dx if dx != 0 else np.zeros_like(x)
    return q - (q[0] + frac * (q[-1] - q[0]))


def geometry_views(configs, views=('side', 'top'), width='full', height=None,
                   annotate=True, shared_transverse=True, chord_relative=True,
                   show_fit=True, mark_w0=True, unit_mm=True, column_headers=True):
    """Paper-style static-geometry panels (one row per configuration).

    The thesis version of ``plotting.plot_initial_geometry``: no titles, no
    summary box, one shared transverse scale so the sag of different cables is
    directly comparable.

    Parameters
    ----------
    configs : sequence of dicts, one per configuration, with keys
        ``label``   short name drawn in the panel corner (e.g. 'C3, 10 cm')
        ``xyz``     (N, 3) gap-sensor positions [m]
        ``cable``   'C1'…'C7'  (colour) — optional if ``color`` is given
        ``fit``     (M, 3) fitted-parabola points [m]        — optional
        ``w0_point``(3,) parabola vertex [m]                 — optional
        ``sag``     midpoint sag [m], used in the annotation — optional
        ``theta``   sag-based Theta, used in the annotation  — optional
        ``color``   explicit colour override                 — optional
    views : any of 'side' (x–z) and 'top' (x–y), in the order to draw them.
    chord_relative : plot the deviation from the straight chord instead of the
        absolute coordinate (default). The mounting tilt is a few millimetres
        while the sag is a few tenths, so the absolute view flattens exactly
        the quantity the figure is about; chord-relative puts both endpoints at
        zero and makes w0 readable. Set False for the raw x–z / x–y geometry.
    shared_transverse : one common transverse range across every panel.
    annotate : draw the label (and w0/Theta if present) in the panel corner.

    Returns (fig, axes) with axes shaped (n_configs, n_views).
    """
    from .config import CABLE_COLORS

    configs = list(configs)
    views = list(views)
    n_rows, n_cols = len(configs), len(views)
    f = 1e3 if unit_mm else 1.0
    unit = 'mm' if unit_mm else 'm'

    h = height if height is not None else 1.15 * n_rows
    w = STYLE['width_full'] if width == 'full' else (
        STYLE['width_half'] if width == 'half' else float(width))
    # constrained layout places the shared supylabel correctly; tight_layout
    # cannot (pass tight=False to save_paper_fig for these figures).
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w, h), squeeze=False,
                             sharex='row', layout='constrained',
                             sharey=True if shared_transverse else 'row')

    def _series(c, j):
        """(x, transverse) traces for one config/view, already scaled."""
        xyz = np.asarray(c['xyz'])
        fit = np.asarray(c['fit']) if c.get('fit') is not None else None
        w0p = np.asarray(c['w0_point']) if c.get('w0_point') is not None else None
        if not chord_relative:
            return (xyz[:, 0] * f, xyz[:, j] * f,
                    None if fit is None else (fit[:, 0] * f, fit[:, j] * f),
                    None if w0p is None else (w0p[0] * f, w0p[j] * f))
        dev = _chord_deviation(xyz[:, 0], xyz[:, j]) * f
        fit_out = w0_out = None
        if fit is not None:
            # Reference the fit to the *sensor* chord so both curves share a datum.
            base = np.interp(fit[:, 0], xyz[[0, -1], 0], xyz[[0, -1], j])
            fit_out = (fit[:, 0] * f, (fit[:, j] - base) * f)
        if w0p is not None:
            base = np.interp(w0p[0], xyz[[0, -1], 0], xyz[[0, -1], j])
            w0_out = (w0p[0] * f, (w0p[j] - base) * f)
        return xyz[:, 0] * f, dev, fit_out, w0_out

    if shared_transverse:
        vals = np.concatenate([np.asarray(_series(c, j)[1]).ravel()
                               for c in configs for j, _, _ in
                               (_GEOM_VIEWS[v] for v in views)])
        span = float(vals.max() - vals.min()) or 1.0
        # Asymmetric padding: the corner annotation lives in the top strip, so
        # keep the data in the lower two thirds of the panel.
        t_lo, t_hi = vals.min() - 0.15 * span, vals.max() + 0.60 * span

    for r, c in enumerate(configs):
        col = c.get('color') or CABLE_COLORS.get(str(c.get('cable')), '#888888')
        for k, view in enumerate(views):
            j, axis_name, header = _GEOM_VIEWS[view]
            ax = axes[r, k]
            x, dev, fit_xy, w0_xy = _series(c, j)

            if chord_relative:
                ax.axhline(0.0, ls='--', color='0.35', lw=0.9, zorder=2,
                           label='chord')
            else:
                ax.plot([x[0], x[-1]], [dev[0], dev[-1]], ls='--', color='0.35',
                        lw=0.9, zorder=2, label='chord')
            ax.plot(x, dev, 'o-', color=col, ms=3.2, lw=0.9, mec='black',
                    mew=0.4, alpha=0.9, zorder=3, label='scan points')
            if show_fit and fit_xy is not None:
                ax.plot(*fit_xy, '-', color='magenta', lw=1.1, alpha=0.85,
                        zorder=4, label='parabola fit')
            if mark_w0 and w0_xy is not None:
                ax.plot(*w0_xy, '*', color='magenta', ms=9, mec='black',
                        mew=0.4, zorder=5, label=r'$w_0$')

            if shared_transverse:
                ax.set_ylim(t_lo, t_hi)
            # With a shared transverse axis one figure-level label serves every
            # panel (set below); otherwise label each row's first column.
            ylabel = None
            if not shared_transverse:
                ylabel = (f'{axis_name} − chord [{unit}]' if chord_relative
                          else f'{axis_name} [{unit}]')
            style_axis(ax, ylabel=ylabel)
            if r == n_rows - 1:
                ax.set_xlabel(f'x [{unit}]')
            if column_headers and r == 0:
                ax.text(0.5, 1.04, header, transform=ax.transAxes,
                        fontsize=STYLE['legend_fs'], va='bottom', ha='center')

        if annotate:
            txt = str(c.get('label', ''))
            if c.get('sag') is not None:
                txt += f"\n$w_0$ = {c['sag'] * 1e3:.2f} mm"
            if c.get('theta') is not None:
                txt += rf",  $\Theta$ = {c['theta']:.3f}"
            axes[r, 0].text(0.015, 0.96, txt, transform=axes[r, 0].transAxes,
                            fontsize=STYLE['legend_fs'], va='top', ha='left',
                            zorder=6)

    if shared_transverse:
        # No explicit x — constrained layout reserves the margin itself, and
        # forcing a position would drop the label on top of the tick labels.
        fig.supylabel(f'deviation from chord [{unit}]' if chord_relative
                      else f'transverse position [{unit}]',
                      fontsize=STYLE['label_fs'])

    # One legend for the whole figure, taken from the first panel.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, -1].legend(handles, labels, fontsize=STYLE['legend_fs'],
                       frameon=False, loc='upper right', ncol=len(labels),
                       handlelength=1.4, columnspacing=0.9,
                       borderpad=0.2, handletextpad=0.4)
    return fig, axes


def wavefield_panels(x, t, components, clim=None, clim_frac=0.9, clim_pct=None,
                     t_lim=None,
                     cbar_label='velocity [mm/s]', scale=1e3, x_in_mm=False,
                     width='full', height=None, annotate=True, cmap='RdBu_r'):
    """Paper-style space–time image panels (red–blue, no titles).

    One row of imshow panels sharing a symmetric colour scale and a single
    colorbar. Time runs downward (seismic-shotgather convention).

    Parameters
    ----------
    x          : (N_s,) sensor / segment positions [m], or a list of such
                 arrays (one per panel — e.g. decimated strain grids).
    t          : (N_t,) time vector [s]
    components : sequence of (label, data) with data (N_t, N_s).
                 label is drawn as a small corner annotation (not a title).
    clim       : symmetric colour limit in *scaled* units; if None, use
                 clim_frac × the max |value| over all panels within t_lim
                 (or clim_frac × the clim_pct-th percentile if clim_pct is
                 set — robust against single broken channels).
    clim_pct   : percentile (e.g. 99.5) used instead of the max when
                 auto-computing clim; None = use the max.
    t_lim      : (t0, t1) time-axis limits [s]; full record if None.
    cbar_label : colorbar label (must match `scale`).
    scale      : multiply data by this before plotting (1e3: m→mm, 1e6: m→µm
                 or strain→µε).
    x_in_mm    : if True, position axis in mm instead of m.
    Returns (fig, axes).
    """
    components = list(components)
    n = len(components)
    xs = x if isinstance(x, (list, tuple)) else [x] * n
    xf = 1e3 if x_in_mm else 1.0

    # constrained layout plays nicer with the shared colorbar than tight_layout
    # (pass tight=False to save_paper_fig when saving these figures).
    fig, axes = paper_figure(1, n, width=width, height=height,
                             sharey=True, squeeze=False, layout='constrained')
    axes = axes[0]

    t0 = t_lim[0] if t_lim is not None else t[0]
    t1 = t_lim[1] if t_lim is not None else t[-1]

    if clim is None:
        tm = (np.asarray(t) >= t0) & (np.asarray(t) <= t1)
        vals = []
        for _, d in components:
            w = np.abs(np.asarray(d)[tm]) * scale
            vals.append(np.nanpercentile(w, clim_pct) if clim_pct is not None
                        else np.nanmax(w))
        clim = clim_frac * max(vals)
    if clim == 0:
        clim = 1e-30

    im = None
    for ax, xi, (label, data) in zip(axes, xs, components):
        im = ax.imshow(np.asarray(data) * scale, aspect='auto', origin='upper',
                       extent=[xi[0] * xf, xi[-1] * xf, t[-1], t[0]],
                       vmin=-clim, vmax=clim, cmap=cmap,
                       interpolation='nearest')
        ax.set_ylim(t1, t0)
        ax.set_xlabel('position [mm]' if x_in_mm else 'position [m]')
        if annotate and label:
            ax.text(0.03, 0.975, label, transform=ax.transAxes,
                    fontsize=STYLE['label_fs'], va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.75, pad=1.5))
    axes[0].set_ylabel('time [s]')

    fig.colorbar(im, ax=axes, label=cbar_label, shrink=0.9, pad=0.02)
    return fig, axes
