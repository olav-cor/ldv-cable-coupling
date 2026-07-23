"""Data loading for cable + shaker .mat files, with kernel-session in-memory cache."""

import numpy as np
import scipy.io as sio
import h5py


# Module-level cache: keyed by cfg['label']. Subsequent loads of the same label
# return the cached arrays without touching disk. Cleared by clear_cache().
_DATASET_CACHE: dict = {}

# Separate cache for cable-free shaker references, keyed by absolute path string.
_SHAKER_REF_CACHE: dict = {}

# Separate cache for single-file line scans (abandoned setups), keyed by
# (path, t_lim, decimate).
_LINE_CACHE: dict = {}


# Fields stored in the cache (everything load_cable_dataset adds to cfg).
_CABLE_CACHE_KEYS = (
    "XYZ", "x", "vx", "vy", "vz",
    "sh_vx", "sh_vy", "sh_vz",
    "t", "fs", "dt", "n_sensors", "n_samples", "shaker_active_idx",
    "idx_left", "idx_right",
)


def clear_cache():
    """Drop all cached datasets and shaker references (frees memory)."""
    _DATASET_CACHE.clear()
    _SHAKER_REF_CACHE.clear()
    _LINE_CACHE.clear()


def _load_mat(filepath):
    """Load a .mat file regardless of version (legacy vs. v7.3/HDF5)."""
    filepath = str(filepath)
    try:
        raw = sio.loadmat(filepath, squeeze_me=True)
        return {k: v for k, v in raw.items() if not k.startswith('_')}
    except NotImplementedError:
        out = {}
        with h5py.File(filepath, 'r') as f:
            for k in f.keys():
                out[k] = np.array(f[k]).T
        return out


def _shape_nt_ns(arr, n_sensors):
    """Reshape an array to (N_t, N_sensors), transposing if MATLAB-style."""
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.shape[0] == n_sensors:
        arr = arr.T
    return arr


def _shape_nt_ns_shaker(arr):
    """Reshape shaker velocity to (N_t, N_sensors).

    We don't know N_sensors a priori; use the heuristic that the time axis
    is the longer one (N_t >> N_sensors).
    """
    arr = np.array(arr)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return arr


def _pick_active_shaker_channel(sh_vx_2d, sh_vy_2d, sh_vz_2d):
    """Return (idx_active, n_alive) for the shaker channel with the most energy."""
    energy = ((sh_vx_2d ** 2).sum(axis=0)
              + (sh_vy_2d ** 2).sum(axis=0)
              + (sh_vz_2d ** 2).sum(axis=0))
    idx_active = int(np.argmax(energy))
    if energy[idx_active] == 0:
        raise ValueError("No active channel found in shaker file.")
    n_alive = int(np.sum(energy > 0))
    return idx_active, n_alive


def load_cable_dataset(cfg, force_reload=False, verbose=True):
    """Populate cfg with loaded cable + shaker arrays.

    After this call cfg has:
        XYZ : (N_s, 3)        coordinates
        x   : (N_s,)          x-coords (sorted)
        vx, vy, vz : (N_t, N_s)  cable velocities  [m/s]
        sh_vx, sh_vy, sh_vz : (N_t,)  shaker velocities (3 components)  [m/s]
        t, fs, dt : time axis
        n_sensors, n_samples, shaker_active_idx

    On second call with the same cfg['label'], returns the cached arrays
    (no disk I/O). Pass force_reload=True to bypass.
    """
    key = cfg['label']
    if not force_reload and key in _DATASET_CACHE:
        cfg.update(_DATASET_CACHE[key])
        return cfg

    # ── Cable ──────────────────────────────────────────────────────────────
    raw = _load_mat(cfg['cable_file'])
    XYZ = np.array(raw['XYZ'])
    if XYZ.shape[0] == 3 and XYZ.shape[1] != 3:
        XYZ = XYZ.T

    ns = XYZ.shape[0]
    vx = _shape_nt_ns(raw['vib_x'], ns)
    vy = _shape_nt_ns(raw['vib_y'], ns)
    vz = _shape_nt_ns(raw['vib_z'], ns)
    t = np.array(raw['t']).ravel()

    # Sort along cable axis (by x)
    sort_idx = np.argsort(XYZ[:, 0])
    XYZ = XYZ[sort_idx]
    vx = vx[:, sort_idx]; vy = vy[:, sort_idx]; vz = vz[:, sort_idx]

    # Drop dead channels (all-zero velocity)
    alive = np.any(vx != 0, axis=0)
    if not alive.all() and verbose:
        print(f"  [{cfg['label']}] removing {(~alive).sum()} dead sensor(s).")
    XYZ = XYZ[alive]; vx = vx[:, alive]; vy = vy[:, alive]; vz = vz[:, alive]

    # Clamp boundary indices in case dead sensor removal shrank the array.
    n_valid = XYZ.shape[0]
    for idx_key, fallback in (('idx_left', 0), ('idx_right', n_valid - 1)):
        orig = cfg.get(idx_key, fallback)
        if orig >= n_valid:
            cfg[idx_key] = n_valid - 1
            if verbose:
                print(f"  [{cfg['label']}] {idx_key} clamped {orig} -> {n_valid - 1} "
                      f"(dead-sensor removal left {n_valid} sensors)")

    fs = 1.0 / (t[1] - t[0])

    # Known shaker offset from base coordinate system
    XYZ[:, 0] -= 0.159
    XYZ[:, 1] -= 0.06
    XYZ[:, 2] -= 0.06

    # ── Shaker reference (single active channel out of many) ──────────────
    rsh = _load_mat(cfg['shaker_file'])
    sh_vx_2d = _shape_nt_ns_shaker(rsh['vib_x'])
    sh_vy_2d = _shape_nt_ns_shaker(rsh['vib_y'])
    sh_vz_2d = _shape_nt_ns_shaker(rsh['vib_z'])

    try:
        idx_active, n_alive = _pick_active_shaker_channel(sh_vx_2d, sh_vy_2d, sh_vz_2d)
    except ValueError as exc:
        raise ValueError(f"[{cfg['label']}] {exc}") from None

    if verbose:
        print(f"  [{cfg['label']}] shaker active channel: idx={idx_active} "
              f"(of {sh_vx_2d.shape[1]} channels; {n_alive} non-zero)")

    sh_vx = sh_vx_2d[:, idx_active]
    sh_vy = sh_vy_2d[:, idx_active]
    sh_vz = sh_vz_2d[:, idx_active]

    # Align lengths
    n_min = min(len(t), len(sh_vx))
    t = t[:n_min]
    vx = vx[:n_min]; vy = vy[:n_min]; vz = vz[:n_min]
    sh_vx = sh_vx[:n_min]; sh_vy = sh_vy[:n_min]; sh_vz = sh_vz[:n_min]

    cfg.update(dict(
        XYZ=XYZ, x=XYZ[:, 0],
        vx=vx, vy=vy, vz=vz,
        sh_vx=sh_vx, sh_vy=sh_vy, sh_vz=sh_vz,
        t=t, fs=fs, dt=1.0 / fs,
        n_sensors=XYZ.shape[0], n_samples=len(t),
        shaker_active_idx=idx_active,
    ))

    # Stash a copy in the cache so the next load is free.
    _DATASET_CACHE[key] = {k: cfg[k] for k in _CABLE_CACHE_KEYS}
    return cfg


def load_line_dataset(path, t_lim=None, decimate=1, lowpass_hz=None,
                      force_reload=False, verbose=True):
    """Load a single line-scan .mat file (no shaker reference, no rig offset).

    For the abandoned-setup recordings (alu beam, ratchet strap, cable
    segment on strap): one file containing XYZ + vib_x/y/z + t. Sensors are
    sorted along x and dead (all-zero) channels removed. Unlike
    load_cable_dataset, no shaker file is expected and no rig coordinate
    offset is applied.

    Parameters
    ----------
    t_lim    : (t0, t1) or None — crop the record to this time window [s]
               before decimation (keeps memory down for long high-fs files).
    decimate : int — keep every k-th sample after zero-phase anti-alias
               filtering (scipy.signal.decimate). 1 = no decimation.
    lowpass_hz : float or None — optional zero-phase Butterworth lowpass
               applied to vx/vy/vz after decimation. Intended for the
               narrow-band transient recordings (e.g. 30 Hz for the 12 Hz
               Ricker files), where everything above the pulse band is
               sensor noise. None = no lowpass.

    Returns dict(XYZ, x, vx, vy, vz, t, fs, dt, n_sensors, n_samples, path).
    """
    key = (str(path), tuple(t_lim) if t_lim is not None else None,
           int(decimate), None if lowpass_hz is None else float(lowpass_hz))
    if not force_reload and key in _LINE_CACHE:
        return dict(_LINE_CACHE[key])

    raw = _load_mat(path)
    XYZ = np.array(raw['XYZ'])
    if XYZ.shape[0] == 3 and XYZ.shape[1] != 3:
        XYZ = XYZ.T

    ns = XYZ.shape[0]
    vx = _shape_nt_ns(raw['vib_x'], ns)
    vy = _shape_nt_ns(raw['vib_y'], ns)
    vz = _shape_nt_ns(raw['vib_z'], ns)
    t = np.array(raw['t']).ravel()

    n_min = min(len(t), vx.shape[0])
    t = t[:n_min]
    vx = vx[:n_min]; vy = vy[:n_min]; vz = vz[:n_min]

    # Sort along the scan line (by x)
    sort_idx = np.argsort(XYZ[:, 0])
    XYZ = XYZ[sort_idx]
    vx = vx[:, sort_idx]; vy = vy[:, sort_idx]; vz = vz[:, sort_idx]

    # Drop dead channels (all-zero velocity)
    alive = np.any(vx != 0, axis=0)
    if not alive.all() and verbose:
        print(f"  [{key[0].rsplit(chr(92), 1)[-1]}] removing "
              f"{(~alive).sum()} dead sensor(s).")
    XYZ = XYZ[alive]; vx = vx[:, alive]; vy = vy[:, alive]; vz = vz[:, alive]

    if t_lim is not None:
        m = (t >= t_lim[0]) & (t <= t_lim[1])
        t = t[m]; vx = vx[m]; vy = vy[m]; vz = vz[m]

    fs = 1.0 / (t[1] - t[0])
    if decimate > 1:
        from scipy.signal import decimate as _dec
        vx = _dec(vx, decimate, axis=0, ftype='fir', zero_phase=True)
        vy = _dec(vy, decimate, axis=0, ftype='fir', zero_phase=True)
        vz = _dec(vz, decimate, axis=0, ftype='fir', zero_phase=True)
        t = t[::decimate][:vx.shape[0]]
        fs = fs / decimate

    if lowpass_hz is not None:
        from .signal import lowpass as _lp
        vx = _lp(vx, fs, lowpass_hz)
        vy = _lp(vy, fs, lowpass_hz)
        vz = _lp(vz, fs, lowpass_hz)

    out = dict(
        XYZ=XYZ, x=XYZ[:, 0],
        vx=vx, vy=vy, vz=vz,
        t=t, fs=fs, dt=1.0 / fs,
        n_sensors=XYZ.shape[0], n_samples=len(t),
        path=key[0],
    )
    _LINE_CACHE[key] = dict(out)
    if verbose:
        lp = '' if lowpass_hz is None else f", lowpass {lowpass_hz:.0f} Hz"
        print(f"  [{key[0].rsplit(chr(92), 1)[-1]}] {out['n_sensors']} sensors, "
              f"{out['n_samples']} samples, fs={fs:.0f} Hz{lp}")
    return out


def load_shaker_reference(path, force_reload=False, verbose=True):
    """Load a cable-free shaker .mat file (e.g. DIRECT_SHAKER_REF_*.mat).

    These files have the same structure as a *_SHAKER.mat (multi-channel
    `vib_x/y/z` arrays) but no `XYZ`. Returns a dict:
        {
            't', 'fs', 'dt',
            'vx', 'vy', 'vz'        # 1-D active channel per component [m/s]
            'shaker_active_idx',
            'n_samples',
            'path',
        }
    """
    key = str(path)
    if not force_reload and key in _SHAKER_REF_CACHE:
        return dict(_SHAKER_REF_CACHE[key])  # shallow copy so callers can't mutate cache

    raw = _load_mat(path)
    sh_vx_2d = _shape_nt_ns_shaker(raw['vib_x'])
    sh_vy_2d = _shape_nt_ns_shaker(raw['vib_y'])
    sh_vz_2d = _shape_nt_ns_shaker(raw['vib_z'])

    idx_active, n_alive = _pick_active_shaker_channel(sh_vx_2d, sh_vy_2d, sh_vz_2d)

    if 't' in raw:
        t = np.array(raw['t']).ravel()
        n_min = min(len(t), sh_vx_2d.shape[0])
        t = t[:n_min]
    else:
        # Reconstruct time axis from sample count assuming fs=5000 Hz default.
        n_min = sh_vx_2d.shape[0]
        t = np.arange(n_min) / 5000.0

    sh_vx_2d = sh_vx_2d[:n_min]
    sh_vy_2d = sh_vy_2d[:n_min]
    sh_vz_2d = sh_vz_2d[:n_min]

    fs = 1.0 / (t[1] - t[0])

    if verbose:
        print(f"  [{key}] shaker active channel idx={idx_active} "
              f"(of {sh_vx_2d.shape[1]} channels; {n_alive} non-zero)  fs={fs:.0f} Hz")

    out = dict(
        t=t, fs=fs, dt=1.0 / fs,
        vx=sh_vx_2d[:, idx_active],
        vy=sh_vy_2d[:, idx_active],
        vz=sh_vz_2d[:, idx_active],
        shaker_active_idx=idx_active,
        n_samples=n_min,
        path=key,
    )
    _SHAKER_REF_CACHE[key] = dict(out)
    return out
