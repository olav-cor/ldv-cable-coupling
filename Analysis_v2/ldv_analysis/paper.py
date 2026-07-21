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
    'marker_size': 5,
    'edge_lw': 0.6,

    # axes cosmetic
    'grid': False,
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
               xscale=None, yscale=None, legend=False, legend_kw=None):
    """Apply the per-axis controls in one call (all optional).

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
    if legend:
        kw = dict(frameon=False)
        kw.update(legend_kw or {})
        ax.legend(**kw)
    return ax


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
