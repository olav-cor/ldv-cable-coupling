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
    modal     - modal analysis (fingerprint, mode shapes, MAC, polarization)
    modal_plotting - modal deep-dive figures, cross-dataset summaries, dashboard
"""

from . import (config, io, signal, geometry, strain, analysis, freqdomain,
               plotting, widgets, modal, modal_plotting)

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
from .freqdomain import (build_coupling_signals, welch_transfer,
                         compute_transfer_functions, band_mean_eta,
                         stft_point_transfer, stft_point_transfers,
                         fit_lorentzian, theoretical_f1,
                         resonance_summary, run_frequency_domain)
from .widgets import strain_comparison_dashboard, linearity_slip_dashboard
from .modal import (analyze_dataset, spectral_fingerprint, detect_resonances,
                    extract_mode_shape, mac, theoretical_modes,
                    clamped_clamped_mode, polarization_ellipse,
                    classify_polarization, dominant_polarization,
                    gap_indices, ALPHA_CC)
from .modal_plotting import (plot_fingerprint, plot_mode_panels, plot_mac_heatmap,
                             plot_ellipse_map, plot_phase_snapshots, deep_dive,
                             build_summary_table, plot_mode_frequency_chart,
                             plot_fn_vs_gap, modal_dashboard)

__all__ = [
    "config", "io", "signal", "geometry", "strain", "analysis", "freqdomain", "plotting", "widgets",
    "modal", "modal_plotting",
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
    "build_coupling_signals", "welch_transfer", "compute_transfer_functions",
    "band_mean_eta", "stft_point_transfer", "stft_point_transfers",
    "fit_lorentzian", "theoretical_f1", "resonance_summary",
    "run_frequency_domain",
    "strain_comparison_dashboard", "linearity_slip_dashboard",
    "analyze_dataset", "spectral_fingerprint", "detect_resonances",
    "extract_mode_shape", "mac", "theoretical_modes", "clamped_clamped_mode",
    "polarization_ellipse", "classify_polarization", "dominant_polarization",
    "gap_indices", "ALPHA_CC",
    "plot_fingerprint", "plot_mode_panels", "plot_mac_heatmap",
    "plot_ellipse_map", "plot_phase_snapshots", "deep_dive",
    "build_summary_table", "plot_mode_frequency_chart", "plot_fn_vs_gap",
    "modal_dashboard",
]
