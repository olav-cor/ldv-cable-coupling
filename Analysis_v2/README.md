# Analysis_v2 — cleaned LDV cable-coupling workspace

Reworked version of the Experiment 2 analysis. The original workspace (parent
folder) is untouched; this folder is self-contained except for the raw data,
which is still read from `../Data` and `../Linearity_Test` (absolute paths in
`ldv_analysis/config.py`).

## Workflow

```
raw .mat data ──► analysis notebooks 01–06 ──► results/*.npz ──► 07_Paper_Figures ──► Figures_paper/
                  (QC plots, titles OK)         (exports)          (no titles, paper style)
```

Each analysis notebook ends with an **export cell** writing a `.npz` to
`results/`. The paper-figure notebook rebuilds every thesis figure from those
files alone — restyling a figure never requires re-running an analysis.

## Notebooks

| Notebook | Purpose | Exports |
|---|---|---|
| `01_Geometry_Sag` | chord, sag (parabola fit + residual), Θ (sag & material), η_pred, sag-vs-theory | `geometry_sag.npz` |
| `02_TimeDomain_QC` | per-frequency window pipeline, 3 strain methods, interactive QC dashboards, thesis example window | `timedomain_example.npz` |
| `03_FrequencyDomain_Coupling` | **main η analysis**: Welch FRFs, H1/H2/Hv/direct estimators, coherence gating, η vs Θ over all configs | `freqdomain_eta.npz`, `freqdomain_frf.npz` |
| `04_Amplitude_Analysis` | displacement/strain amplitudes, shaker transfer, **extension/compression asymmetry** | `amplitude_reference.npz`, `asymmetry.npz` |
| `05_Modal_Analysis` | Welch-smoothed resonance detection, MAC, polarization, **shaker input-bias diagnostics** | `modal_summary.npz` + `.csv` |
| `06_Linearity_Analysis` | η vs drive amplitude, FD cross-check per segment, ext/comp per amplitude bin, slip dashboards | `linearity_eta.npz` |
| `07_Paper_Figures` | final thesis figures from `results/` only | `Figures_paper/*.png/.pdf` |

## Package (`ldv_analysis/`)

| Module | Responsibility | Changed vs v1 |
|---|---|---|
| `config.py` | catalogue, paths, sweep params, uncertainties | + `RESULTS_DIR`, `FIG_DIR`, `GAP_LENGTH_STD_M` |
| `io.py` | .mat loading + cache | unchanged |
| `signal.py` | bandpass, FFT integration, spectra | unchanged |
| `geometry.py` | chord, sag, projection | + `sag_fit_diagnostics` (residual → sag uncertainty) |
| `strain.py` | 3 strain estimators | unchanged |
| `analysis.py` | time-domain QC + linearity pipeline | − `compute_mean_eta`, `spectral_peak_summary` (deprecated); + sag residual in `prepare_geometry` |
| `freqdomain.py` | Welch FRFs, coherence, η(f) | + `welch_spectra`, `estimate_H` (**H1/H2/Hv**), `direct_transfer`; `compute_transfer_functions(estimator=…)` takes `'h1'/'h2'/'hv'` or `'direct'`/`None` (un-averaged Y(f)/X(f), dense grid) and stores all variants; `frf_arrays`/`band_mean_frf` are grid-aware |
| `amplitude.py` | amplitude sweeps | unchanged |
| `asymmetry.py` | **new** — extension vs compression (half-cycle η, peak asymmetry, even harmonics) | new |
| `modal.py` | modal pipeline | + Welch/smoothed fingerprint, `shaker_input_spectrum`, input-normalized detection |
| `modal_plotting.py` | modal figures | + `plot_shaker_directionality` |
| `plotting.py` | QC/overview figures | − time-domain η summaries & spectral-peak plots; + wrapped phase (`phase_mode`), `plot_estimator_comparison`, asymmetry plots |
| `widgets.py` | interactive dashboards | unchanged |
| `export.py` | **new** — standardized `.npz` exports | new |
| `paper.py` | **new** — paper figure style/save helpers (no titles, size/xlim/ylim control) | new |

## Removed from the daily workflow (still in the v1 folder)

- Time-domain band-mean η (`compute_mean_eta`) and its η-vs-Θ plot → replaced
  by the coherence-gated FRF band mean.
- `LDV_EtaTheta_Analysis` notebook → sag comparison moved to 01, η vs Θ to 03.
- `LDV_Spectral_Analysis` notebook (not needed at the moment).

## Thesis material

`docs/processing_chapter.tex` — full "Signal Processing" chapter draft with
all derivations, matching the code equation-by-equation (each equation has a
comment naming the implementing function).
