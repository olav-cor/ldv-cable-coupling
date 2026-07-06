"""Standardised .npz export/import of analysis results.

The analysis notebooks (01–06) end with an export cell that calls
:func:`export_results`; the paper-figure notebook (07_Paper_Figures) rebuilds
every thesis figure from these files alone — no reprocessing, no dependence on
the kernel state of the analysis notebooks (the pattern of
``paper_plots_from_exports.ipynb``).

Conventions
-----------
* One .npz per analysis topic, e.g. ``results/freqdomain_eta.npz``.
* Per-dataset scalar quantities are stored as aligned 1-D arrays plus a
  ``labels`` string array, so a row is recovered by index.
* Per-dataset curves (e.g. FRFs) are stored as 2-D arrays (n_datasets, n_f)
  with a shared ``f`` axis, NaN-padded if lengths differ.
* Every file carries ``meta`` (a JSON string) with the parameters used, so a
  figure caption can always be reconstructed from the file itself.
"""

import json
import datetime

import numpy as np

from .config import RESULTS_DIR


def export_results(name, meta=None, **arrays):
    """Save arrays to results/<name>.npz with a JSON meta record.

    meta    : dict of parameters (JSON-serialisable) — timestamp is added.
    arrays  : name=array pairs; lists of strings are converted to str arrays,
              lists of arrays with unequal length are NaN-padded to 2-D.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.npz"

    out = {}
    for key, val in arrays.items():
        if isinstance(val, (list, tuple)) and len(val) and isinstance(val[0], str):
            out[key] = np.asarray(val, dtype=str)
        elif (isinstance(val, (list, tuple)) and len(val)
              and isinstance(val[0], np.ndarray)):
            out[key] = _pad_stack(val)
        else:
            out[key] = np.asarray(val)

    meta = dict(meta or {})
    meta['exported'] = datetime.datetime.now().isoformat(timespec='seconds')
    out['meta'] = np.asarray(json.dumps(meta, default=str))

    np.savez_compressed(path, **out)
    print(f"exported {path.name}: "
          + ", ".join(f"{k}{v.shape}" for k, v in out.items() if k != 'meta'))
    return path


def _pad_stack(arr_list):
    """Stack 1-D arrays of unequal length into a NaN-padded 2-D array."""
    n = max(len(a) for a in arr_list)
    out = np.full((len(arr_list), n), np.nan)
    for i, a in enumerate(arr_list):
        out[i, :len(a)] = a
    return out


def load_results(name):
    """Load results/<name>.npz → dict of arrays (+ 'meta' parsed to a dict)."""
    path = RESULTS_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run the corresponding analysis notebook's "
            "export cell first.")
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    if 'meta' in out:
        out['meta'] = json.loads(str(out['meta']))
    return out


def list_exports():
    """List available export files with their meta timestamps."""
    if not RESULTS_DIR.exists():
        print(f"(no exports yet — {RESULTS_DIR} is empty)")
        return []
    files = sorted(RESULTS_DIR.glob('*.npz'))
    for p in files:
        try:
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
            print(f"  {p.name:35s} exported {meta.get('exported', '?')}")
        except Exception as exc:                       # unreadable file
            print(f"  {p.name:35s} <unreadable: {exc}>")
    return files
