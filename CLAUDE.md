# CLAUDE.md — LDV Cable-Coupling Experiment 2

## Project context

Master thesis experiment at IMENSUS GmbH. The goal is to quantify how well vibrational energy couples from a shaker into cables of different geometry, measured with a Laser Doppler Vibrometer (LDV).

## Code layout

All reusable logic lives in the `ldv_analysis/` package. The notebooks import from it and should stay thin (visualisation + narrative only).

| Module | Responsibility |
|---|---|
| `config.py` | Dataset catalogue (`ALL_DATASETS`), absolute data paths, sweep parameters, plot styling |
| `io.py` | Load `.mat` files; `load_cable_dataset()` and `load_shaker_reference()` with LRU cache |
| `signal.py` | `bandpass`, `integrate_fft`, `amplitude_spectrum`, `compute_spectrogram`, `find_dominant_peaks` |
| `geometry.py` | `compute_chord`, `measure_sag`, `project_onto_chord` |
| `strain.py` | `strain_spatial_gradient`, `strain_3d_arclength` |
| `analysis.py` | `per_frequency_qc`, `spectral_peak_summary`, `prepare_geometry`, `linearity_qc` |
| `plotting.py` | All figure-generation functions (space-time, QC dashboard, spectra, …) |
| `widgets.py` | `strain_comparison_dashboard` — ipywidgets interactive browser |

## Data paths

Data paths are **hardcoded absolute Windows paths** in `config.py`:

```python
BASE    = Path(r"C:\Users\ocornelius\OneDrive - IMENSUS GmbH\Dokumente\Master_Thesis\Experiment2\Data")
LIN_BASE = Path(r"C:\Users\ocornelius\OneDrive - IMENSUS GmbH\Dokumente\Master_Thesis\Experiment2\Linearity_Test")
```

If running on a different machine, update these two variables. The data files are `.mat` (MATLAB) and `.svd` format, 9–17 MB each. They are not tracked in git.

## Dataset naming convention

```
Cable{N}_{gap}cm[_{condition}]_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_{speed}mms_m2
```

- `N` — cable number 1–7
- `gap` — 5, 10, or 15 cm mounting gap
- `condition` — omitted (taut), `Sag`, `1mmSag`, `2mmSag`, `3mmSag`, or `TAPE` (Cable7)
- `_SHAKER` suffix — shaker reference channel only

## Key parameters (config.py)

- Sweep: 1–500 Hz, 12.5 s, log sweep
- Sample rate: 5 kHz, 60 000 samples per measurement
- Bandpass: fractional bandwidth `BANDWIDTH_FRAC` around target frequency, order `BP_ORDER`
- Integration: high-pass at `INT_FMIN_HP` Hz before amplitude extraction

## Working conventions

- Keep notebooks thin; move reusable logic into `ldv_analysis/`.
- `__pycache__/` and `.ipynb_checkpoints/` are gitignored — never commit them.
- Generated figures go to `Figures/` (gitignored); re-run the notebook to reproduce them.
- The `clear_cache()` function in `io.py` frees the in-memory `.mat` cache if memory is tight.
