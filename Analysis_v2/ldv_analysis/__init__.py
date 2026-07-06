"""LDV cable-coupling analysis utilities — Analysis_v2 (cleaned workspace).

Submodules:
    config    - dataset catalogue, paths, sweep timing, uncertainties, styling
    io        - .mat loading (with in-memory cache), shaker-reference loader
    signal    - bandpass, FFT integration, amplitude spectrum, spectrogram
    geometry  - chord, sag (+ parabola-fit diagnostics), projection onto chord
    strain    - three strain estimators (gradient, arc-length, Fourier)
    analysis  - time-domain per-frequency QC + linearity pipeline
    freqdomain- Welch FRFs with H1/H2/Hv/direct estimators, coherence, η(f)
    amplitude - displacement / strain amplitude overview across the swept band
    asymmetry - extension vs compression response asymmetry
    modal     - modal analysis (smoothed fingerprint, MAC, polarization,
                shaker input-bias diagnostics)
    modal_plotting - modal figures + shaker directionality plot
    plotting  - QC / overview figures (notebook use; titles allowed)
    widgets   - ipywidgets-based interactive browsers
    export    - standardized .npz result exports (feeds 07_Paper_Figures)
    paper     - publication-figure style + helpers (no titles, LaTeX-ready)
"""

from . import (config, io, signal, geometry, strain, analysis, freqdomain,
               amplitude, asymmetry, plotting, widgets, modal, modal_plotting,
               export, paper)

# Convenience re-exports
from .config import (
    ALL_DATASETS, BASE, LIN_BASE, RESULTS_DIR, FIG_DIR, TARGET_FREQS,
    SWEEP_BUFFER_S, SWEEP_F_START, SWEEP_F_END, SWEEP_DURATION,
    N_CYCLES_PER_WINDOW, BANDWIDTH_FRAC, BP_ORDER, INT_FMIN_HP,
    GAP_LENGTH_STD_M, SENSOR_POS_STD_M,
    CABLE_COLORS, GAP_MARKERS,
    LIN_FREQUENCIES, LIN_BUFFER_DURATION, LIN_SEGMENT_DURATION, LIN_FADE_DURATION,
    LINEARITY_LABELS, linearity_segment_window,
    SWEEP_LABELS,
    CABLE_PROPERTIES, cable_section_AI, theta_from_sag, theta_from_material,
    dataset_style, sweep_time_of_frequency, select_datasets,
)
from .io import load_cable_dataset, load_shaker_reference, clear_cache
from .signal import (bandpass, integrate_fft, amplitude_spectrum,
                     compute_spectrogram, find_dominant_peaks)
from .geometry import (compute_chord, measure_sag, sag_fit_diagnostics,
                       project_onto_chord)
from .strain import (strain_spatial_gradient, strain_3d_arclength,
                     strain_fourier_gradient)
from .analysis import (per_frequency_qc, prepare_geometry,
                       linearity_qc, run_linearity_analysis)
from .freqdomain import (build_coupling_signals,
                         welch_spectra, estimate_H, welch_transfer,
                         direct_transfer,
                         compute_transfer_functions, band_mean_eta,
                         stft_point_transfer, stft_point_transfers,
                         fit_lorentzian, theoretical_f1,
                         resonance_summary, run_frequency_domain)
from .amplitude import (amplitude_at_frequency, amplitude_sweep,
                        band_mean_amplitude)
from .asymmetry import (asymmetry_at_frequency, asymmetry_sweep,
                        linearity_segment_asymmetry)
from .widgets import (strain_comparison_dashboard, linearity_slip_dashboard,
                      amplitude_browser)
from .modal import (analyze_dataset, spectral_fingerprint, smooth_spectrum,
                    shaker_input_spectrum, detect_resonances,
                    extract_mode_shape, mac, theoretical_modes,
                    clamped_clamped_mode, polarization_ellipse,
                    classify_polarization, dominant_polarization,
                    gap_indices, ALPHA_CC)
from .modal_plotting import (plot_fingerprint, plot_mode_panels, plot_mac_heatmap,
                             plot_ellipse_map, plot_phase_snapshots, deep_dive,
                             build_summary_table, plot_mode_frequency_chart,
                             plot_fn_vs_gap, modal_dashboard,
                             plot_shaker_directionality)
from .export import export_results, load_results, list_exports

__all__ = [
    "config", "io", "signal", "geometry", "strain", "analysis", "freqdomain",
    "amplitude", "asymmetry", "plotting", "widgets",
    "modal", "modal_plotting", "export", "paper",
    "ALL_DATASETS", "BASE", "LIN_BASE", "RESULTS_DIR", "FIG_DIR", "TARGET_FREQS",
    "SWEEP_BUFFER_S", "SWEEP_F_START", "SWEEP_F_END", "SWEEP_DURATION",
    "N_CYCLES_PER_WINDOW", "BANDWIDTH_FRAC", "BP_ORDER", "INT_FMIN_HP",
    "GAP_LENGTH_STD_M", "SENSOR_POS_STD_M",
    "CABLE_COLORS", "GAP_MARKERS",
    "dataset_style", "sweep_time_of_frequency", "select_datasets",
    "load_cable_dataset", "load_shaker_reference", "clear_cache",
    "bandpass", "integrate_fft", "amplitude_spectrum", "compute_spectrogram",
    "find_dominant_peaks",
    "compute_chord", "measure_sag", "sag_fit_diagnostics", "project_onto_chord",
    "strain_spatial_gradient", "strain_3d_arclength", "strain_fourier_gradient",
    "per_frequency_qc", "prepare_geometry", "linearity_qc",
    "run_linearity_analysis",
    "build_coupling_signals", "welch_spectra", "estimate_H", "welch_transfer",
    "direct_transfer", "compute_transfer_functions", "band_mean_eta",
    "stft_point_transfer", "stft_point_transfers",
    "fit_lorentzian", "theoretical_f1", "resonance_summary",
    "run_frequency_domain",
    "amplitude_at_frequency", "amplitude_sweep", "band_mean_amplitude",
    "asymmetry_at_frequency", "asymmetry_sweep", "linearity_segment_asymmetry",
    "strain_comparison_dashboard", "linearity_slip_dashboard", "amplitude_browser",
    "analyze_dataset", "spectral_fingerprint", "smooth_spectrum",
    "shaker_input_spectrum", "detect_resonances",
    "extract_mode_shape", "mac", "theoretical_modes", "clamped_clamped_mode",
    "polarization_ellipse", "classify_polarization", "dominant_polarization",
    "gap_indices", "ALPHA_CC",
    "plot_fingerprint", "plot_mode_panels", "plot_mac_heatmap",
    "plot_ellipse_map", "plot_phase_snapshots", "deep_dive",
    "build_summary_table", "plot_mode_frequency_chart", "plot_fn_vs_gap",
    "modal_dashboard", "plot_shaker_directionality",
    "export_results", "load_results", "list_exports",
]
