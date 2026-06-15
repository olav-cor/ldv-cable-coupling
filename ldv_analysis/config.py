"""Dataset catalogue, file paths, sweep parameters, and plot styling."""

from pathlib import Path
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Base directories
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\ocornelius\OneDrive - IMENSUS GmbH\Dokumente\Master_Thesis\Experiment2\Data")
LIN_BASE = Path(r"C:\Users\ocornelius\OneDrive - IMENSUS GmbH\Dokumente\Master_Thesis\Experiment2\Linearity_Test")

# Cable-free shaker reference (sweep only, no cable connected)
DIRECT_SHAKER_REF = BASE / "DIRECT_SHAKER_REF_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat"


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
CABLE_COLORS = {
    "Cable1": "#404040",
    "Cable2": "#E600FF",
    "Cable3": "#0022FF",
    "Cable4": "#FCF800",
    "Cable5": "#FF0000",
    "Cable6": "#09FF05",
    "Cable7": "#1ABCAC",
}

GAP_MARKERS = {
    0.05: "D",
    0.10: "s",
    0.15: "o",
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
ALL_DATASETS = [

    # ── Cable 1 (D = 3.8 mm → r = 1.9 mm) ───────────────────────────────────
    dict(label="Cable1_5cm",
         cable_file=BASE / "Cable1_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable1_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable1", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.0019),
    dict(label="Cable1_5cm_Sag",
         cable_file=BASE / "Cable1_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable1_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable1", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.0019),
    dict(label="Cable1_10cm",
         cable_file=BASE / "Cable1_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable1_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable1", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="Cable1_10cm_Sag",
         cable_file=BASE / "Cable1_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable1_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable1", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="Cable1_15cm",
         cable_file=BASE / "Cable1_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable1_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable1", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="Cable1_15cm_Sag",
         cable_file=BASE / "IS_Cable1_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "IS_Cable1_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable1", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),

    # ── Cable 2 (D = 3.8 mm → r = 1.9 mm) ───────────────────────────────────
    dict(label="Cable2_5cm",
         cable_file=BASE / "Cable2_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable2_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable2", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.0019),
    dict(label="Cable2_5cm_Sag",
         cable_file=BASE / "Cable2_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable2_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable2", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.0019),
    dict(label="Cable2_10cm",
         cable_file=BASE / "Cable2_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable2_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable2", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="Cable2_10cm_Sag",
         cable_file=BASE / "Cable2_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable2_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable2", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="Cable2_15cm",
         cable_file=BASE / "Cable2_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable2_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable2", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),
    dict(label="Cable2_15cm_Sag",
         cable_file=BASE / "IS_Cable2_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "IS_Cable2_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable2", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0019),

    # ── Cable 3 (flat ribbon 1.9 x 3.1 mm, r ≈ 0.95 mm) ──────────────────────────
    dict(label="Cable3_5cm",
         cable_file=BASE / "Cable3_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable3_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable3", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00095),
    dict(label="Cable3_5cm_Sag",
         cable_file=BASE / "Cable3_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable3_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable3", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00095),
    dict(label="Cable3_10cm",
         cable_file=BASE / "Cable3_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable3_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="Cable3", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00095),
    dict(label="Cable3_10cm_Sag",
         cable_file=BASE / "Cable3_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable3_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="Cable3", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00095),
    dict(label="Cable3_15cm",
         cable_file=BASE / "Cable3_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable3_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable3", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00095),
    dict(label="Cable3_15cm_Sag",
         cable_file=BASE / "IS_Cable3_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "IS_Cable3_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="Cable3", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00095),

    # ── Cable 4 (D = 2.3 mm → r = 1.15 mm) ───────────────────────────────────
    dict(label="Cable4_5cm",
         cable_file=BASE / "Cable4_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable4_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable4", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00115),
    dict(label="Cable4_5cm_Sag",
         cable_file=BASE / "Cable4_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable4_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable4", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00115),
    dict(label="Cable4_10cm",
         cable_file=BASE / "Cable4_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable4_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="Cable4", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00115),
    dict(label="Cable4_10cm_b",  # repeat recording at 500 mm/s LDV range
         cable_file=BASE / "Cable4_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable4_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable4", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00115),
    dict(label="Cable4_10cm_Sag",
         cable_file=BASE / "Cable4_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable4_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="Cable4", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00115),
    dict(label="Cable4_15cm",
         cable_file=BASE / "Cable4_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable4_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable4", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00115),
    dict(label="Cable4_15cm_Sag",
         cable_file=BASE / "IS_Cable4_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "IS_Cable4_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="Cable4", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00115),

    # ── Cable 5 (D = 5.5 mm → r = 2.75 mm) ──────────────────────────────────
    dict(label="Cable5_5cm",
         cable_file=BASE / "Cable5_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable5_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable5", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00275),
    dict(label="Cable5_5cm_Sag",
         cable_file=BASE / "Cable5_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable5_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable5", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00275),
    dict(label="Cable5_10cm",
         cable_file=BASE / "Cable5_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable5_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable5", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00275),
    dict(label="Cable5_10cm_Sag",
         cable_file=BASE / "Cable5_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable5_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable5", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00275),
    dict(label="Cable5_15cm",
         cable_file=BASE / "Cable5_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable5_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable5", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00275),
    dict(label="Cable5_15cm_IS",
         cable_file=BASE / "IS_Cable5_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "IS_Cable5_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable5", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00275),
    dict(label="Cable5_15cm_Sag",
         cable_file=BASE / "IS_Cable5_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "IS_Cable5_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable5", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00275),

    # ── Cable 6 (D = 3.0 mm → r = 1.5 mm) ───────────────────────────────────
    dict(label="Cable6_5cm",
         cable_file=BASE / "Cable6_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable6", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.0015),
    dict(label="Cable6_5cm_Sag",
         cable_file=BASE / "Cable6_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable6", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.0015),
    dict(label="Cable6_10cm",
         cable_file=BASE / "Cable6_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable6", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0015),
    dict(label="Cable6_10cm_Sag",
         cable_file=BASE / "Cable6_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable6", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0015),
    dict(label="Cable6_15cm",
         cable_file=BASE / "Cable6_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable6", gap_m=0.15, idx_left=0, idx_right=15, shaker_end="right", radius_m=0.0015),
    dict(label="Cable6_15cm_Sag",
         cable_file=BASE / "Cable6_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2.mat",
         shaker_file=BASE / "Cable6_15cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable6", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.0015),

    # ── Cable 7 (D = 0.9 mm → r = 0.45 mm) ──────────────────────────────────
    # Note: only one SHAKER file exists for 5cm (labelled "Sag" — used for both)
    dict(label="Cable7_5cm",
         cable_file=BASE / "Cable7_5cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable7_5cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="Cable7", gap_m=0.05, idx_left=0, idx_right=8, shaker_end="right", radius_m=0.00045),
    dict(label="Cable7_10cm",
         cable_file=BASE / "Cable7_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable7_10cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="Cable7", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),
    dict(label="Cable7_10cm_Sag",
         cable_file=BASE / "Cable7_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
         shaker_file=BASE / "Cable7_10cm_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
         cable="Cable7", gap_m=0.10, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),
    # Note: cable file uses 5000 mm/s range; shaker file uses 500 mm/s range
    dict(label="Cable7_15cm_NO_Tape",
         cable_file=BASE / "Cable7_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_5000mms_m2.mat",
         shaker_file=BASE / "Cable7_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2_SHAKER.mat",
         cable="Cable7", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),
    dict(label="Cable7_15cm",
         cable_file=BASE / "Cable7_15cm_TAPE_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_5000mms_m2.mat",
         shaker_file=BASE / "Cable7_15cm_TAPE_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_5000mms_m2_SHAKER.mat",
         cable="Cable7", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),
    dict(label="Cable7_15cm_Sag",
         cable_file=BASE / "Cable7_15cm_TAPE_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_5000mms_m2.mat",
         shaker_file=BASE / "Cable7_15cm_TAPE_Sag_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_5000mms_m2_SHAKER.mat",
         cable="Cable7", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),
    # Cable7_15cm_Sag is the IS rig setup with the 1250 mm/s range
    #dict(label="Cable7_15cm_Sag",
    #     cable_file=BASE / "IS_Cable7_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2.mat",
    #     shaker_file=BASE / "IS_Cable7_15cm_avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2_SHAKER.mat",
    #     cable="Cable7", gap_m=0.15, idx_left=0, idx_right=16, shaker_end="right", radius_m=0.00045),

    # ── Linearity tests (190k samples ≈ 38 s — different sweep timing) ──────
    dict(label="LinTest_Cable5_5cm",
         cable_file=LIN_BASE / "Cable5_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2.mat",
         shaker_file=LIN_BASE / "Cable5_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2_SHAKER.mat",
         cable="Cable5", gap_m=0.05, idx_left=0, idx_right=2, shaker_end="right", radius_m=0.00275),
    dict(label="LinTest_Cable6_5cm",
         cable_file=LIN_BASE / "Cable6_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2.mat",
         shaker_file=LIN_BASE / "Cable6_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2_SHAKER.mat",
         cable="Cable6", gap_m=0.05, idx_left=0, idx_right=2, shaker_end="right", radius_m=0.0015),
    dict(label="LinTest_Cable7_5cm",
         cable_file=LIN_BASE / "Cable7_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2.mat",
         shaker_file=LIN_BASE / "Cable7_5cm_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2_SHAKER.mat",
         cable="Cable7", gap_m=0.05, idx_left=0, idx_right=2, shaker_end="right", radius_m=0.00045),
    dict(label="LinTest_Cable7_10cm_Sag",
         cable_file=LIN_BASE / "Cable7_10cm_Sag_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2.mat",
         shaker_file=LIN_BASE / "Cable7_10cm_Sag_avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2_SHAKER.mat",
         cable="Cable7", gap_m=0.10, idx_left=0, idx_right=2, shaker_end="right", radius_m=0.00045),
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
# Cable3 is a flat ribbon (2 mm × 3.05 mm).  When lying flat, gravity bends it
# in the 2 mm (height) direction → bending_axis='weak'.
# ─────────────────────────────────────────────────────────────────────────────
CABLE_PROPERTIES = {
    "Cable1": dict(cross_section='circular',    radius_m=0.0019,
                   rho=2300.21, E=51.98 * 1e9),
    "Cable2": dict(cross_section='circular',    radius_m=0.0019,
                   rho=1196.97, E=None),
    "Cable3": dict(cross_section='rectangular', width_m=0.0031, height_m=0.0019,
                   bending_axis='weak',
                   rho=1562.65, E=None),
    "Cable4": dict(cross_section='circular',    radius_m=0.00115,
                   rho=1179.37, E=None),
    "Cable5": dict(cross_section='circular',    radius_m=0.00275,
                   rho=1050.08, E=13.34 * 1e9),
    "Cable6": dict(cross_section='circular',    radius_m=0.0015,
                   rho=904.424, E=1.33 * 1e9),
    "Cable7": dict(cross_section='circular',    radius_m=0.00045,
                   rho=1571.90, E=None),
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


def theta_from_material(cable_name, L, g=9.81):
    """Compute Θ from physical parameters (Probst et al., Eq. 14).

    Θ = ρ²g²A³L⁸ / (128π⁸E²I³)

    L    : segment span [m] — use chord_len from prepare_geometry.
    g    : gravitational acceleration [m/s²] (default 9.81).

    Returns None if rho or E is not yet set in CABLE_PROPERTIES.
    """
    props = CABLE_PROPERTIES.get(cable_name)
    if props is None:
        return None
    rho = props.get('rho')
    E = props.get('E')
    if rho is None or E is None:
        return None
    A, I = cable_section_AI(cable_name)
    return rho ** 2 * g ** 2 * A ** 3 * L ** 8 / (128 * np.pi ** 8 * E ** 2 * I ** 3)


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

low_freqs  = np.arange(2,   100,  6) # From, Too, Step
mid_freqs  = np.arange(100, 250, 10)
high_freqs = np.arange(250, 500, 25)
TARGET_FREQS = np.unique(np.concatenate((low_freqs, mid_freqs, high_freqs))).tolist()



N_CYCLES_PER_WINDOW = 10
BANDWIDTH_FRAC = 0.5

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
    "LinTest_Cable5_5cm",
    "LinTest_Cable6_5cm",
    "LinTest_Cable7_5cm",
    "LinTest_Cable7_10cm_Sag",
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
