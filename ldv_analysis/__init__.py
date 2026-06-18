"""LDV cable-coupling analysis utilities.

Submodules:
    config    - dataset catalogue, paths, sweep timing, plot styling
    io        - .mat loading (with in-memory cache), shaker-reference loader
    signal    - bandpass, FFT integration, amplitude spectrum, spectrogram
    geometry  - chord, sag, projection onto chord
    strain    - spatial-gradient strain and 3-D arc-length per-segment strain
    analysis  - per-frequency QC pipeline
    plotting  - space-time, QC dashboard, raw-trace, spectra, spectrogram plots
    widgets   - ipywidgets-based interactive browsers
"""

from . import config, io, signal, geometry, strain, analysis, plotting, widgets

# Convenience re-exports
from .config import (
    ALL_DATASETS, BASE, LIN_BASE, TARGET_FREQS,
    SWEEP_BUFFER_S, SWEEP_F_START, SWEEP_F_END, SWEEP_DURATION,
    N_CYCLES_PER_WINDOW, BANDWIDTH_FRAC, BP_ORDER, INT_FMIN_HP,
    CABLE_COLORS, GAP_MARKERS,
    LIN_FREQUENCIES, LIN_BUFFER_DURATION, LIN_SEGMENT_DURATION, LIN_FADE_DURATION,
    LINEARITY_LABELS, linearity_segment_window,
    SWEEP_LABELS,
    CABLE_PROPERTIES, cable_section_AI, theta_from_sag, theta_from_material,
    dataset_style, sweep_time_of_frequency, select_datasets,
)
from .io import load_cable_dataset, load_shaker_reference, clear_cache
from .signal import bandpass, integrate_fft, amplitude_spectrum, compute_spectrogram, find_dominant_peaks
from .geometry import compute_chord, measure_sag, project_onto_chord
from .strain import strain_spatial_gradient, strain_3d_arclength
from .analysis import (per_frequency_qc, spectral_peak_summary,
                       prepare_geometry, linearity_qc, run_linearity_analysis,
                       compute_mean_eta)
from .widgets import strain_comparison_dashboard, linearity_slip_dashboard

__all__ = [
    "config", "io", "signal", "geometry", "strain", "analysis", "plotting", "widgets",
    "ALL_DATASETS", "BASE", "LIN_BASE", "TARGET_FREQS",
    "SWEEP_BUFFER_S", "SWEEP_F_START", "SWEEP_F_END", "SWEEP_DURATION",
    "N_CYCLES_PER_WINDOW", "BANDWIDTH_FRAC", "BP_ORDER", "INT_FMIN_HP",
    "CABLE_COLORS", "GAP_MARKERS",
    "dataset_style", "sweep_time_of_frequency", "select_datasets",
    "load_cable_dataset", "load_shaker_reference", "clear_cache",
    "bandpass", "integrate_fft", "amplitude_spectrum", "compute_spectrogram", "find_dominant_peaks",
    "compute_chord", "measure_sag", "project_onto_chord",
    "strain_spatial_gradient", "strain_3d_arclength",
    "per_frequency_qc", "spectral_peak_summary",
    "strain_comparison_dashboard", "linearity_slip_dashboard",
]
