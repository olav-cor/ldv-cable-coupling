"""Dataset catalogue, file paths, sweep parameters, and plot styling."""

from pathlib import Path
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Base directories
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\ocornelius\OneDrive - IMENSUS GmbH\Dokumente\Master_Thesis\Experiment2\Data")
LIN_BASE = Path(r"C:\Users\ocornelius\OneDrive - IMENSUS GmbH\Dokumente\Master_Thesis\Experiment2\Linearity_Test")

# Analysis_v2 output locations (relative to this package's parent folder).
# RESULTS_DIR holds the .npz exports the paper-figure notebook reads;
# FIG_DIR holds the final thesis figures.
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = Path(__file__).resolve().parents[1] / "Figures_paper"

# ─────────────────────────────────────────────────────────────────────────────
# Measurement uncertainties (used for error bars in paper figures)
# ─────────────────────────────────────────────────────────────────────────────
# Reading error of the gap length between the mounting points (ruler placement).
GAP_LENGTH_STD_M = 0.005      # ± 0.5 cm
# LDV sensor-position uncertainty entering the sag measurement; the parabola
# fit additionally returns an RMS residual per dataset (geometry.measure_sag).
SENSOR_POS_STD_M = 0.0005     # ± 0.5 mm nominal scan-point placement

# Cable-free shaker reference (sweep only, no cable connected)
DIRECT_SHAKER_REF = BASE / "DIRECT_SHAKER_REF_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat"


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
CABLE_COLORS = {
    "C1": "#FF0000", 
    "C2": "#404040",
    "C3": "#E600FF",
    "C4": "#0022FF",
    "C5": "#09FF05",
    "C6": "#FCF800",
    "C7": "#1ABCAC",
}

# CABLE_COLORS = {
#     "Cable1": "#404040",
#     "Cable2": "#E600FF",
#     "Cable3": "#0022FF",
#     "Cable4": "#FCF800",
#     "Cable5": "#FF0000",
#     "Cable6": "#09FF05",
#     "Cable7": "#1ABCAC",
# }

# Segment length
GAP_MARKERS = {
    0.05: "o",
    0.10: "s",
    0.15: "D",
    0.20: "X",
}


def dataset_style(cfg):
    """Return (color, marker, fillstyle) for a dataset.

    fillstyle is 'bottom' for Sag variants, 'full' otherwise.
    """
    col = CABLE_COLORS.get(cfg.get('cable', ''), '#888888')
    marker = GAP_MARKERS.get(round(cfg['gap_m'], 2), 's')
    fillstyle = 'none' if 'Sag' in cfg.get('label', '') else 'full'
    return col, marker, fillstyle


# ─────────────────────────────────────────────────────────────────────────────
# Master dataset catalogue
#
#   cable      : cable identifier key → colour via CABLE_COLORS
#   gap_m      : nominal gap [m]      → marker via GAP_MARKERS
#   idx_left/right : sensor indices spanning the hanging gap (verify after loading)
#   shaker_end : "right" or "left"
#   radius_m   : effective cable radius for theoretical Θ
# ─────────────────────────────────────────────────────────────────────────────

# WARNING THE ID NUMBERING OF THE CABLES WAS CHANGED AND THEREFORE THE DATASETS HAVE DIFFERENT LABELS, DO NOT CHANGE BACK
#
# The cable IDs used everywhere in the analysis and in the thesis figures are
# C1…C7.  The raw .mat/.svd filenames still carry the OLD acquisition numbering
# and must NOT be renamed.  Mapping (label/cable  ←  filename):
#
#   C1 ← Cable5      C2 ← Cable1      C3 ← Cable2      C4 ← Cable3
#   C5 ← Cable6      C6 ← Cable4      C7 ← Cable7
#
# So a dataset labelled "C1_10cm" legitimately loads "Cable5_10cm_*.mat".
ALL_DATASETS = [

    # ── C2  [data files: Cable1]  (D = 3.8 mm → r = 1.9 mm) ─────────────────
    dict(label="C2_5cm",
         cable_file=BASE / "Cable1_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable1_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C2", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.0019),
    dict(label="C2_5cm_Sag",
         cable_file=BASE / "Cable1_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable1_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C2", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.0019),
    dict(label="C2_10cm",
         cable_file=BASE / "Cable1_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable1_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C2", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="C2_10cm_Sag",
         cable_file=BASE / "Cable1_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable1_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C2", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="C2_15cm",
         cable_file=BASE / "Cable1_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable1_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C2", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="C2_15cm_Sag",
         cable_file=BASE / "IS_Cable1_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "IS_Cable1_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C2", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),

    # ── C3  [data files: Cable2]  (D = 3.8 mm → r = 1.9 mm) ─────────────────
    dict(label="C3_5cm",
         cable_file=BASE / "Cable2_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable2_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C3", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.0019),
    dict(label="C3_5cm_Sag",
         cable_file=BASE / "Cable2_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable2_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C3", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.0019),
    dict(label="C3_10cm",
         cable_file=BASE / "Cable2_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable2_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C3", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="C3_10cm_Sag",
         cable_file=BASE / "Cable2_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable2_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C3", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="C3_15cm",
         cable_file=BASE / "Cable2_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable2_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C3", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="C3_15cm_Sag",
         cable_file=BASE / "IS_Cable2_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "IS_Cable2_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C3", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),

    # ── C4  [data files: Cable3]  (flat ribbon 1.9 x 3.1 mm, r ≈ 0.95 mm) ───
    dict(label="C4_5cm",
         cable_file=BASE / "Cable3_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable3_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C4", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00095),
    dict(label="C4_5cm_Sag",
         cable_file=BASE / "Cable3_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable3_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C4", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00095),
    dict(label="C4_10cm",
         cable_file=BASE / "Cable3_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable3_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="C4", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00095),
    dict(label="C4_10cm_Sag",
         cable_file=BASE / "Cable3_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable3_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="C4", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00095),
    dict(label="C4_15cm",
         cable_file=BASE / "Cable3_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable3_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C4", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00095),
    dict(label="C4_15cm_Sag",
         cable_file=BASE / "IS_Cable3_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "IS_Cable3_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="C4", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00095),

    # ── C6  [data files: Cable4]  (D = 2.3 mm → r = 1.15 mm) ────────────────
    dict(label="C6_5cm",
         cable_file=BASE / "Cable4_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable4_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C6", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00115),
    dict(label="C6_5cm_Sag",
         cable_file=BASE / "Cable4_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable4_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C6", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00115),
    dict(label="C6_10cm",
         cable_file=BASE / "Cable4_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable4_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="C6", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00115),
    dict(label="C6_10cm_b",  # repeat recording at 500 mm/s LDV range
         cable_file=BASE / "Cable4_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable4_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C6", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00115),
    dict(label="C6_10cm_Sag",
         cable_file=BASE / "Cable4_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable4_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="C6", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00115),
    dict(label="C6_15cm",
         cable_file=BASE / "Cable4_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable4_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C6", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00115),
    dict(label="C6_15cm_Sag",
         cable_file=BASE / "IS_Cable4_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "IS_Cable4_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="C6", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00115),

    # ── C1  [data files: Cable5]  (D = 5.5 mm → r = 2.75 mm) ────────────────
    dict(label="C1_5cm",
         cable_file=BASE / "Cable5_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable5_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C1", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00275),
    dict(label="C1_5cm_Sag",
         cable_file=BASE / "Cable5_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable5_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C1", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00275),
    dict(label="C1_10cm",
         cable_file=BASE / "Cable5_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable5_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C1", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00275),
    dict(label="C1_10cm_Sag",
         cable_file=BASE / "Cable5_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable5_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C1", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00275),
#     dict(label="C1_15cm",
#          cable_file=BASE / "Cable5_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
#          shaker_file=BASE / "Cable5_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
#          cable="C1", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00275),
     dict(label="C1_15cm",
         cable_file=BASE / "IS_Cable5_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "IS_Cable5_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C1", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00275),
    dict(label="C1_15cm_Sag",
         cable_file=BASE / "IS_Cable5_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "IS_Cable5_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C1", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00275),

    # ── C5  [data files: Cable6]  (D = 2.9 mm → r = 1.45 mm) ────────────────
    dict(label="C5_5cm",
         cable_file=BASE / "Cable6_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C5", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00145),
    dict(label="C5_5cm_Sag",
         cable_file=BASE / "Cable6_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C5", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00145),
    dict(label="C5_10cm",
         cable_file=BASE / "Cable6_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C5", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00145),
    dict(label="C5_10cm_Sag",
         cable_file=BASE / "Cable6_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C5", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00145),
    dict(label="C5_15cm",
         cable_file=BASE / "Cable6_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C5", gap_m=0.15, idx_left=0, idx_right=15, shaker_end="right", radius_m=0.00145),
    dict(label="C5_15cm_Sag",
         cable_file=BASE / "Cable6_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="C5", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00145),

    # ── C7  [data files: Cable7]  (D = 0.9 mm → r = 0.45 mm) ────────────────
    # Note: only one SHAKER file exists for 5cm (labelled "Sag" — used for both)
    dict(label="C7_5cm",
         cable_file=BASE / "Cable7_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable7_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="C7", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00045),
    dict(label="C7_10cm",
         cable_file=BASE / "Cable7_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable7_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="C7", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),
    dict(label="C7_10cm_Sag",
         cable_file=BASE / "Cable7_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable7_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="C7", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),
    # Note: cable file uses 5000 mm/s range; shaker file uses 500 mm/s range
 #   dict(label="C7_15cm_NO_Tape",
 #        cable_file=BASE / "Cable7_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_5000mms_m2.mat",
 #        shaker_file=BASE / "Cable7_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
 #        cable="C7", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),
    dict(label="C7_15cm",
         cable_file=BASE / "Cable7_15cm_TAPE_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_5000mms_m2.mat",
         shaker_file=BASE / "Cable7_15cm_TAPE_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_5000mms_m2_SHAKER.mat",
         cable="C7", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),
    dict(label="C7_15cm_Sag",
         cable_file=BASE / "Cable7_15cm_TAPE_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_5000mms_m2.mat",
         shaker_file=BASE / "Cable7_15cm_TAPE_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_5000mms_m2_SHAKER.mat",
         cable="C7", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),
    # C7_15cm_Sag is the IS rig setup with the 1250 mm/s range
    #dict(label="C7_15cm_Sag",
    #     cable_file=BASE / "IS_Cable7_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
    #     shaker_file=BASE / "IS_Cable7_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
    #     cable="C7", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),

    # ── Linearity tests (190k samples ≈ 38 s — different sweep timing) ──────
    dict(label="LinTest_C1_5cm",
         cable_file=LIN_BASE / "Cable5_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2.mat",
         shaker_file=LIN_BASE / "Cable5_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2_SHAKER.mat",
         cable="C1", gap_m=0.05, idx_left=0, idx_right=2, shaker_end="right", radius_m=0.00275),
    dict(label="LinTest_C5_5cm",
         cable_file=LIN_BASE / "Cable6_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2.mat",
         shaker_file=LIN_BASE / "Cable6_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2_SHAKER.mat",
         cable="C5", gap_m=0.05, idx_left=0, idx_right=2, shaker_end="right", radius_m=0.00145),
    dict(label="LinTest_C7_5cm",
         cable_file=LIN_BASE / "Cable7_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2.mat",
         shaker_file=LIN_BASE / "Cable7_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2_SHAKER.mat",
         cable="C7", gap_m=0.05, idx_left=0, idx_right=2, shaker_end="right", radius_m=0.00045),
    dict(label="LinTest_C7_10cm_Sag",
         cable_file=LIN_BASE / "Cable7_10cm_Sag_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2.mat",
         shaker_file=LIN_BASE / "Cable7_10cm_Sag_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2_SHAKER.mat",
         cable="C7", gap_m=0.10, idx_left=0, idx_right=2, shaker_end="right", radius_m=0.00045),
]


SWEEP_LABELS = [ds['label'] for ds in ALL_DATASETS if not ds['label'].startswith('LinTest')]


# ─────────────────────────────────────────────────────────────────────────────
# Cable physical properties
#
# Fill in rho [kg/m³] and E [Pa] once measured.  Until then, leave as None —
# any function that needs them will return None rather than crash.
#
# Cross-section geometry:
#   circular    → radius_m [m]
#   rectangular → width_m, height_m [m]
#                 bending_axis: 'weak'   means I = width * height³ / 12
#                                        (height is the dimension in the gravity
#                                         direction — use for a ribbon lying flat)
#                               'strong' means I = height * width³ / 12
#
# C4 is a flat ribbon (2 mm × 3.05 mm).  When lying flat, gravity bends it
# in the 2 mm (height) direction → bending_axis='weak'.
# ─────────────────────────────────────────────────────────────────────────────
CABLE_PROPERTIES = {
    "C2": dict(cross_section='circular',    radius_m=0.0019,
                   rho=2265.20, E=[24.54 * 1e9, 30.95 * 1e9, 30.38 * 1e9]),  # three values from three different tests
    "C3": dict(cross_section='circular',    radius_m=0.0019,
                   rho=1156.85, E=[12.0 * 1e9, 12.29 * 1e9, 10.11 * 1e9]),  # three values from three different tests
    "C4": dict(cross_section='rectangular', width_m=0.0031, height_m=0.0019,
                   bending_axis='weak',
                   rho=1534.80, E=[4.14 * 1e9, 4.02 * 1e9, 4.46 * 1e9]),  # three values from three different tests
    "C6": dict(cross_section='circular',    radius_m=0.00115,
                   rho=1191.41, E=[6.52 * 1e9, 7.34 * 1e9, 6.5 * 1e9]),  # three values from three different tests
    "C1": dict(cross_section='circular',    radius_m=0.00275,
                   rho=1051.42, E=[9.74 * 1e9, 8.27 * 1e9, 8.61 * 1e9]),  # three values from three different tests
    "C5": dict(cross_section='circular',    radius_m=0.00145,
                   rho=961.36, E=[3.03 * 1e9, 2.74 * 1e9, 2.61 * 1e9]),  # three values from three different tests
    "C7": dict(cross_section='circular',    radius_m=0.00045,
                   rho=1241.80, E=[2.08 * 1e9, 2.25 * 1e9, 1.85 * 1e9]),  # three values from three different tests
}


def cable_section_AI(cable_name):
    """Return (A [m²], I [m⁴]) for a named cable from CABLE_PROPERTIES.

    For circular cables  : A = π r², I = π r⁴ / 4.
    For rectangular cables: A = w·h,  I = w·h³/12 ('weak') or h·w³/12 ('strong'),
                            where 'weak' = bending in the height direction (gravity).
    """
    props = CABLE_PROPERTIES.get(cable_name)
    if props is None:
        raise KeyError(f"Cable '{cable_name}' not in CABLE_PROPERTIES")
    if props['cross_section'] == 'circular':
        r = props['radius_m']
        return np.pi * r ** 2, np.pi * r ** 4 / 4
    else:
        w = props['width_m']
        h = props['height_m']
        A = w * h
        if props.get('bending_axis', 'weak') == 'weak':
            I = w * h ** 3 / 12
        else:
            I = h * w ** 3 / 12
        return A, I


def _E_stats(cable_name):
    """Return (E_mean [Pa], E_std [Pa]) for a cable.

    E in CABLE_PROPERTIES may be a scalar or a list of repeated measurements.
    Returns (None, None) if E is not set.
    """
    E = CABLE_PROPERTIES.get(cable_name, {}).get('E')
    if E is None:
        return None, None
    if isinstance(E, (list, tuple)):
        arr = np.asarray(E, dtype=float)
        return float(arr.mean()), float(arr.std())
    return float(E), 0.0


def theta_from_material(cable_name, L, g=9.81):
    """Compute Θ from physical parameters (Probst et al., Eq. 14).

    Θ = ρ²g²A³L⁸ / (128π⁸E²I³)

    L    : segment span [m] — use chord_len from prepare_geometry.
    g    : gravitational acceleration [m/s²] (default 9.81).

    If E in CABLE_PROPERTIES is a list of repeated measurements, Θ is computed
    individually for each measurement and the sample mean and std are returned.

    Returns
    -------
    (theta_mean, theta_std) — both None if rho or E is not set.
    theta_std is 0.0 when E is a single scalar value.
    """
    props = CABLE_PROPERTIES.get(cable_name)
    if props is None:
        return None, None
    rho = props.get('rho')
    E_raw = props.get('E')
    if rho is None or E_raw is None:
        return None, None
    A, I = cable_section_AI(cable_name)
    C = rho ** 2 * g ** 2 * A ** 3 * L ** 8 / (128 * np.pi ** 8 * I ** 3)
    if isinstance(E_raw, (list, tuple)):
        thetas = np.asarray([C / e ** 2 for e in E_raw])
        return float(thetas.mean()), float(thetas.std())
    return float(C / E_raw ** 2), 0.0


def theta_from_sag(cable_name, sag):
    """Compute Θ from measured midpoint sag (Probst et al., general form).

    Θ = A·w₀² / (8·I)

    For circular sections this is equivalent to (1/2)(w₀/r)².
    For rectangular sections it correctly accounts for the non-circular geometry.

    sag : midpoint sag w₀ [m].
    Returns None if cable_name is not in CABLE_PROPERTIES.
    """
    if cable_name not in CABLE_PROPERTIES:
        return None
    A, I = cable_section_AI(cable_name)
    return A * sag ** 2 / (8 * I)


def select_datasets(active_labels):
    """Return the subset of ALL_DATASETS whose labels appear in active_labels (order preserved).

    Prints a warning for any labels not found in ALL_DATASETS.
    """
    label_map = {ds['label']: ds for ds in ALL_DATASETS}
    out = [label_map[lbl] for lbl in active_labels if lbl in label_map]
    missing = [lbl for lbl in active_labels if lbl not in label_map]
    if missing:
        print(f"WARNING - labels not found in ALL_DATASETS: {missing}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sweep timing (must match Sweep_Generation.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_BUFFER_S = 1.0
SWEEP_F_START = 1.0
SWEEP_F_END = 500.0
SWEEP_DURATION = 10.0


def sweep_time_of_frequency(f):
    """Time [s] at which a log-sweep reaches instantaneous frequency f [Hz]."""
    return SWEEP_BUFFER_S + SWEEP_DURATION * np.log(f / SWEEP_F_START) / np.log(SWEEP_F_END / SWEEP_F_START)


# ─────────────────────────────────────────────────────────────────────────────
# Target frequencies for per-frequency analysis
# ─────────────────────────────────────────────────────────────────────────────
# TARGET_FREQS = [5, 20, 50, 100, 150, 200, 300, 400]

# low_freqs  = np.arange(2,   100,  6) # From, Too, Step
low_freqs = np.arange(2,100, 2)
mid_freqs  = np.arange(100, 250, 10)
high_freqs = np.arange(250, 500, 25)
TARGET_FREQS = np.unique(np.concatenate((low_freqs, mid_freqs, high_freqs))).tolist()



N_CYCLES_PER_WINDOW = 5 # 10
BANDWIDTH_FRAC = 0.5 # 0.5
# Alternative: fixed absolute bandwidth in Hz (±N Hz around f_target).
# Set to a float (e.g. 5.0) to use instead of BANDWIDTH_FRAC.
# None means use the fractional mode.
BANDWIDTH_ABS_HZ = 5

# Filter / integration safety
BP_ORDER = 4
INT_FMIN_HP = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Linearity test signal parameters (must match Sweep_Generation.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
LIN_FREQUENCIES = [5, 20, 50, 100, 150, 200, 300, 400]   # Hz
LIN_BUFFER_DURATION = 1.0    # s — leading silence and gap between segments
LIN_SEGMENT_DURATION = 3.5   # s — duration of each mono-frequency segment
LIN_FADE_DURATION = 0.5      # s — linear amplitude fade at end of each segment

LINEARITY_LABELS = [
    "LinTest_C1_5cm",
    "LinTest_C5_5cm",
    "LinTest_C7_5cm",
    "LinTest_C7_10cm_Sag",
]


def linearity_segment_window(f_target, frequencies=None, buffer=None, segment=None):
    """Return (t_start, t_end) [s] for f_target in the linearity source signal."""
    if frequencies is None:
        frequencies = LIN_FREQUENCIES
    if buffer is None:
        buffer = LIN_BUFFER_DURATION
    if segment is None:
        segment = LIN_SEGMENT_DURATION
    idx = list(frequencies).index(f_target)
    t_start = buffer + idx * (segment + buffer)
    return t_start, t_start + segment
