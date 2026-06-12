# LDV Cable-Coupling Analysis — Experiment 2

Master thesis experiment: characterising the vibrational coupling efficiency of cables of varying geometry using a Laser Doppler Vibrometer (LDV).

## Overview

Seven cables (different diameters and materials) were excited with a log-sweep shaker (1–500 Hz). The LDV scanned the cable at multiple sensor positions. Coupling efficiency is evaluated as a function of:

- **Cable identity** — Cable1 through Cable7
- **Gap distance** — 5 cm, 10 cm, 15 cm
- **Sag condition** — taut / natural sag / forced sag
- **Tape reinforcement** (Cable7 only)

The `ldv_analysis` Python package provides all signal processing, geometry computation, and plotting used in the Jupyter notebooks.

## Repository structure

```
Experiment2/
├── ldv_analysis/          # Python package (signal, geometry, strain, plotting, …)
│   ├── config.py          # dataset catalogue, file paths, sweep parameters
│   ├── io.py              # .mat loader with in-memory cache
│   ├── signal.py          # bandpass, FFT integration, amplitude spectrum
│   ├── geometry.py        # chord, sag, projection onto chord
│   ├── strain.py          # spatial-gradient and arc-length strain
│   ├── analysis.py        # per-frequency QC pipeline
│   ├── plotting.py        # all figure-generation functions
│   └── widgets.py         # ipywidgets interactive browsers
│
├── LDV_Cable_Coupling_Analysis.ipynb   # main coupling efficiency analysis
├── LDV_Cable_Coupling_v2.ipynb         # revised coupling analysis
├── LDV_EtaTheta_Analysis.ipynb         # η–Θ relationship
├── LDV_Linearity_Analysis.ipynb        # linearity / amplitude sweep
├── LDV_Spectral_Analysis.ipynb         # spectral content
└── Sweep_Generation.ipynb              # source sweep signal generation
```

**Not tracked in git** (stored locally on OneDrive):
- `Data/` — raw `.mat`/`.svd` measurement files (~several GB)
- `Linearity_Test/` — linearity test data
- `Figures/` — generated PNG outputs
- `Videos_Fotos/` — experiment videos and photos

## Setup

```python
# The data paths in ldv_analysis/config.py are absolute Windows paths.
# Adjust BASE and LIN_BASE if running on a different machine.
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `ipywidgets`, `jupyter`

```bash
pip install numpy scipy matplotlib ipywidgets jupyter
```
