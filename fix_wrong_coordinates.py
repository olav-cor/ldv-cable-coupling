"""Fix LDV datasets recorded in the wrong coordinate system.

On 2026-07-14 a subset of measurements was recorded without the user-defined
3-D alignment loaded, so the PSV stored scan-point positions (XYZ) *and*
velocity components (vib_x/y/z) in the scanner's native frame instead of the
rig frame used by every other dataset.  Those files live in
Data/Wrong_Coordinates/.

Because the LDV head and the rig did not move, a single rigid transform
(rotation R + translation t) maps the wrong frame onto the correct one:

    XYZ_correct = R @ XYZ_wrong + t          (positions)
    v_correct   = R @ v_wrong                (velocity vectors, per sensor)

R and t are fitted (weighted Kabsch) from physical points that were measured
in BOTH frames on the same rig:

  * the shaker scan point — every *_SHAKER.mat contains the cable points plus
    one extra point on the shaker itself (identified as the strong outlier
    from the cable line).  It is ~19 mm off the cable axis, so it also pins
    the roll angle about the cable axis.
  * the mount-side cable endpoints of same-gap datasets that exist in the
    correct frame (20 cm gaps of Cables 2/3/4/6 recorded the same evening;
    the older Cable5_5cm_Sag and Cable7_5cm references).

The script prints fit residuals and several physical sanity checks (sag must
dip in -z, the dominant vibration component must become vib_z), then writes
corrected copies — same filenames — into the normal Data folder.  The raw
originals in Wrong_Coordinates/ and the .svd files are left untouched.

Usage:
    python fix_wrong_coordinates.py            # fit, validate, write
    python fix_wrong_coordinates.py --dry-run  # fit + validate only
    python fix_wrong_coordinates.py --overwrite  # allow replacing existing outputs
"""

import argparse
import sys

import numpy as np
import scipy.io as sio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ldv_analysis.config import BASE, LIN_BASE
from ldv_analysis.geometry import measure_sag

WRONG_DIR = BASE / "Wrong_Coordinates"

S500 = "avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_500mms_m2"
S1250 = "avg5_05V_R2_fs5kHz_60000nt_LogSweep_1_500Hz_12_5s_1250mms_m2"
SLIN = "avg5_05V_R2_fs5kHz_190000nt_Lin_Test_Hz_40s_1250mms_m2"

# Log-sweep datasets → Data folder
WRONG_MATS = [
    f"Cable1_20cm_{S500}.mat",
    f"Cable1_20cm_{S500}_SHAKER.mat",
    f"Cable1_20cm_Sag_{S500}.mat",
    f"Cable1_20cm_Sag_{S500}_SHAKER.mat",
    f"Cable5_20cm_{S500}.mat",
    f"Cable5_20cm_{S500}_SHAKER.mat",
    f"Cable5_20cm_Sag_{S500}.mat",
    f"Cable5_20cm_Sag_{S500}_SHAKER.mat",
    f"Cable5_5cm_SagB_{S500}.mat",
    f"Cable5_5cm_SagB_{S500}_SHAKER.mat",
    f"Cable7_5cm_Sag_{S1250}.mat",  # no SHAKER partner (Data already has one)
]

# Linearity-test datasets → Linearity_Test folder
WRONG_LIN_MATS = [
    f"Cable5_5cm_SagB_{SLIN}.mat",
    f"Cable5_5cm_SagB_{SLIN}_SHAKER.mat",
    f"Cable7_5cm_Sag_{SLIN}.mat",
    f"Cable7_5cm_Sag_{SLIN}_SHAKER.mat",
]

# (source filename, destination directory) for everything to convert
MANIFEST = [(f, BASE) for f in WRONG_MATS] + [(f, LIN_BASE) for f in WRONG_LIN_MATS]

# Wrong-frame SHAKER files used to locate the shaker point in the wrong frame
WRONG_SHAKER_MATS = [f for f, _ in MANIFEST if f.endswith("_SHAKER.mat")]

# Correct-frame SHAKER files from the same evening (20 cm session)
REF_SHAKER_MATS = (
    [f"Cable{n}_20cm_{S500}_SHAKER.mat" for n in (2, 3, 4, 6)]
    + [f"Cable{n}_20cm_Sag_{S500}_SHAKER.mat" for n in (2, 3, 4, 6)]
    + [f"Cable7_20cm_{S1250}_SHAKER.mat", f"Cable7_20cm_Sag_{S1250}_SHAKER.mat"]
)

# Correct-frame taut 20 cm cable files → endpoint targets for the wrong 20 cm sets
REF_20CM_TAUT = [f"Cable{n}_20cm_{S500}.mat" for n in (2, 3, 4, 6)]


# ────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ────────────────────────────────────────────────────────────────────────────
def load_xyz(path):
    d = sio.loadmat(path, squeeze_me=True)
    X = np.asarray(d["XYZ"], dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[0] == 3 and X.shape[1] != 3:
        X = X.T
    return X


def line_fit(X):
    """Best-fit line through points: (centroid, unit direction)."""
    c = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - c)
    return c, Vt[0]


def perp_dist(X, c, e):
    d = X - c
    return np.linalg.norm(d - np.outer(d @ e, e), axis=1)


def shaker_point(path):
    """Extract the shaker scan point from a *_SHAKER.mat.

    The file holds the cable scan points plus one point on the shaker; that
    point is a strong outlier (>5 mm) from the cable line while the cable
    points scatter <3 mm around it.
    """
    X = load_xyz(path)
    # The shaker point is the one whose removal leaves the straightest line
    # through the remaining (cable) points.  Trying every removal is robust
    # even for the 3+1-point linearity files, where a PCA line through all
    # points is dragged towards the shaker point.
    best = None
    for i in range(len(X)):
        rest = np.delete(np.arange(len(X)), i)
        c2, e2 = line_fit(X[rest])
        score = perp_dist(X[rest], c2, e2).max()
        if best is None or score < best[1]:
            best = (i, score, c2, e2)
    i, d_cable, c2, e2 = best
    d_shaker = perp_dist(X[i : i + 1], c2, e2)[0]
    # cable points may deviate a few mm from a straight line (sag); the shaker
    # point must be both far off the line and clearly separated from that scatter
    if not (d_shaker > 0.008 and d_shaker > 3.0 * d_cable):
        raise RuntimeError(
            f"Could not isolate shaker point in {path} "
            f"(outlier {d_shaker*1e3:.1f} mm, cable scatter {d_cable*1e3:.1f} mm)"
        )
    return X[i]


def endpoints(X, shaker_pt):
    """(near-shaker endpoint, far endpoint) of a cable point set."""
    c, e = line_fit(X)
    s = (X - c) @ e
    p_lo, p_hi = X[np.argmin(s)], X[np.argmax(s)]
    if np.linalg.norm(p_lo - shaker_pt) <= np.linalg.norm(p_hi - shaker_pt):
        return p_lo, p_hi
    return p_hi, p_lo


def weighted_kabsch(P, Q, w):
    """Rigid transform (R, t) minimising Σ wᵢ‖R·Pᵢ + t − Qᵢ‖²."""
    w = np.asarray(w, dtype=float)
    p0 = (w[:, None] * P).sum(axis=0) / w.sum()
    q0 = (w[:, None] * Q).sum(axis=0) / w.sum()
    H = (P - p0).T @ np.diag(w) @ (Q - q0)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q0 - R @ p0
    return R, t


# ────────────────────────────────────────────────────────────────────────────
# Fit the transform
# ────────────────────────────────────────────────────────────────────────────
def fit_transform():
    print("── Fitting rigid transform (wrong frame → correct frame) ──")

    # Shaker point in both frames (mean over files)
    sh_w = np.mean([shaker_point(WRONG_DIR / f) for f in WRONG_SHAKER_MATS], axis=0)
    sh_r = np.mean([shaker_point(BASE / f) for f in REF_SHAKER_MATS], axis=0)
    print(f"shaker point  wrong: {np.round(sh_w, 4)}   correct: {np.round(sh_r, 4)}")

    pairs = [("shaker point", sh_w, sh_r, 4.0)]

    # 20 cm mount endpoints: wrong taut Cable1/Cable5 ↔ mean of same-evening
    # correct taut 20 cm datasets (Cables 2/3/4/6)
    ref20 = [endpoints(load_xyz(BASE / f), sh_r) for f in REF_20CM_TAUT]
    ref20_near = np.mean([p for p, _ in ref20], axis=0)
    ref20_far = np.mean([q for _, q in ref20], axis=0)
    for lbl in ("Cable1_20cm", "Cable5_20cm"):
        near, far = endpoints(load_xyz(WRONG_DIR / f"{lbl}_{S500}.mat"), sh_w)
        pairs.append((f"{lbl} near end", near, ref20_near, 1.0))
        pairs.append((f"{lbl} far end", far, ref20_far, 1.0))

    # 5 cm mount endpoints ↔ older correct 5 cm references (same cable each)
    for w_file, r_file, lbl in [
        (f"Cable5_5cm_SagB_{S500}.mat", f"Cable5_5cm_Sag_{S500}.mat", "Cable5_5cm"),
        (f"Cable7_5cm_Sag_{S1250}.mat", f"Cable7_5cm_{S1250}.mat", "Cable7_5cm"),
    ]:
        w_near, w_far = endpoints(load_xyz(WRONG_DIR / w_file), sh_w)
        r_near, r_far = endpoints(load_xyz(BASE / r_file), sh_r)
        pairs.append((f"{lbl} near end", w_near, r_near, 1.0))
        pairs.append((f"{lbl} far end", w_far, r_far, 1.0))

    P = np.array([p for _, p, _, _ in pairs])
    Q = np.array([q for _, _, q, _ in pairs])
    w = np.array([wt for _, _, _, wt in pairs])
    R, t = weighted_kabsch(P, Q, w)

    print(f"\nR (det = {np.linalg.det(R):+.6f}):\n{np.round(R, 5)}")
    print(f"t = {np.round(t, 5)}")

    res = np.linalg.norm((P @ R.T + t) - Q, axis=1)
    print("\nresiduals:")
    for (lbl, *_), r_mm in zip(pairs, res * 1e3):
        print(f"  {lbl:22s} {r_mm:6.2f} mm")
    rms = np.sqrt(np.average(res**2, weights=w)) * 1e3
    print(f"  weighted RMS          {rms:6.2f} mm")
    if res.max() > 0.005:
        raise RuntimeError("Fit residual exceeds 5 mm — correspondences look wrong.")
    return R, t, sh_w


# ────────────────────────────────────────────────────────────────────────────
# Physical sanity checks
# ────────────────────────────────────────────────────────────────────────────
def validate(R, t, sh_w):
    print("\n── Validation ──")
    ok = True

    # 1) Sag must dip towards -z after the transform
    for f in [
        f"Cable1_20cm_Sag_{S500}.mat",
        f"Cable5_20cm_Sag_{S500}.mat",
        f"Cable5_5cm_SagB_{S500}.mat",
        f"Cable7_5cm_Sag_{S1250}.mat",
    ]:
        X = load_xyz(WRONG_DIR / f) @ R.T + t
        X = X[np.argsort(X[:, 0])]
        sag, idx, _ = measure_sag(X, 0, len(X) - 1)
        c, e = line_fit(X)
        dev = (X[idx] - X[0]) - ((X[idx] - X[0]) @ e) * e
        dev_hat = dev / np.linalg.norm(dev)
        ang = np.degrees(np.arccos(np.clip(-dev_hat[2], -1, 1)))
        flag = "OK" if ang < 30 else "FAIL"
        ok &= ang < 30
        print(f"  {f.split('_avg5')[0]:22s} sag {sag*1e3:5.2f} mm, "
              f"dip direction {np.round(dev_hat, 3)} ({ang:4.1f}° from -z)  [{flag}]")

    # 2) Taut 20 cm cable must land on the rig line (y ≈ 0.059, z ≈ 0.063)
    for f in [f"Cable1_20cm_{S500}.mat", f"Cable5_20cm_{S500}.mat"]:
        X = load_xyz(WRONG_DIR / f) @ R.T + t
        print(f"  {f.split('_avg5')[0]:22s} x [{X[:,0].min():.4f}, {X[:,0].max():.4f}]  "
              f"y {X[:,1].mean():.4f}±{X[:,1].std()*1e3:.1f}mm  "
              f"z {X[:,2].mean():.4f}±{X[:,2].std()*1e3:.1f}mm")

    # 3) After rotation the dominant vibration component must be vib_z
    d = sio.loadmat(WRONG_DIR / f"Cable5_5cm_SagB_{S500}.mat", squeeze_me=True)
    V = np.stack([d["vib_x"], d["vib_y"], d["vib_z"]])
    Vn = np.einsum("ij,j...->i...", R, V)
    rms = np.sqrt((Vn**2).mean(axis=(1, 2)))
    flag = "OK" if rms[2] == rms.max() else "FAIL"
    ok &= rms[2] == rms.max()
    print(f"  velocity RMS after rotation (Cable5_5cm_SagB): "
          f"vx {rms[0]:.4f}  vy {rms[1]:.4f}  vz {rms[2]:.4f}  [{flag}]")

    if not ok:
        raise RuntimeError("Validation failed — not writing any files.")


# ────────────────────────────────────────────────────────────────────────────
# Convert files
# ────────────────────────────────────────────────────────────────────────────
def convert_file(src, dst, R, t):
    d = sio.loadmat(src)  # no squeeze: preserve shapes for a faithful re-save
    out = {k: v for k, v in d.items() if not k.startswith("__")}

    X = np.asarray(out["XYZ"], dtype=float)
    transposed = X.shape[0] == 3 and X.shape[1] != 3
    if transposed:
        X = X.T
    Xn = X @ R.T + t
    out["XYZ"] = Xn.T if transposed else Xn

    V = np.stack([out["vib_x"], out["vib_y"], out["vib_z"]])  # (3, ns, nt)
    Vn = np.einsum("ij,j...->i...", R, V)
    out["vib_x"], out["vib_y"], out["vib_z"] = Vn[0], Vn[1], Vn[2]

    sio.savemat(dst, out, do_compression=True)

    # Round-trip verification
    chk = sio.loadmat(dst)
    assert np.allclose(chk["XYZ"], out["XYZ"]), dst
    for k in ("vib_x", "vib_y", "vib_z"):
        assert np.allclose(chk[k], out[k]), (dst, k)
    assert np.array_equal(chk["t"], d["t"]), dst
    if "ref1" in d:
        assert np.array_equal(chk["ref1"], d["ref1"]), dst
    missing = set(k for k in d if not k.startswith("__")) - set(
        k for k in chk if not k.startswith("__"))
    assert not missing, (dst, missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="fit and validate only, write nothing")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing existing output files")
    ap.add_argument("--skip-existing", action="store_true",
                    help="convert only files whose output does not exist yet")
    args = ap.parse_args()

    R, t, sh_w = fit_transform()
    validate(R, t, sh_w)

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    todo = MANIFEST
    if args.skip_existing:
        todo = [(f, d) for f, d in MANIFEST if not (d / f).exists()]
        skipped = len(MANIFEST) - len(todo)
        if skipped:
            print(f"\n--skip-existing: {skipped} already-converted file(s) skipped.")

    conflicts = [f for f, d in todo if (d / f).exists()]
    if conflicts and not args.overwrite:
        print("\nRefusing to overwrite existing files (use --overwrite or --skip-existing):")
        for f in conflicts:
            print(f"  {f}")
        sys.exit(1)

    print("\n── Writing corrected files ──")
    for f, dst_dir in todo:
        convert_file(WRONG_DIR / f, dst_dir / f, R, t)
        print(f"  ✓ {f}  →  {dst_dir.name}/")
    print(f"\nDone: {len(todo)} files corrected and verified. "
          f"Originals remain in {WRONG_DIR}.")


if __name__ == "__main__":
    main()
