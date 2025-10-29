#!/usr/bin/env python3
"""
make_stamps_dataset.py  —  test-only candidate stamp builder

Creates 5-channel stamps (C,H,W) = [F1, F2, F3, S2, S3] for each candidate:
  S2 = F2 - median(F1, F3)
  S3 = F3 - median(F1, F2)

Writes (per folder):
  <out_root>/<folder>/test/*.npz
  <out_root>/<folder>/metadata.csv
    columns: id,split,folder,cx,cy,label=-1,source,stamp_path,f1_x..f3_y

Expectations:
  - <root_dir>/<folder>/aligned/ contains 3 aligned FITS
  - candidates_csv has rows with: folder, and either (f2_x,f2_y) or (f1_x,f1_y)
  - No dependency on truth or is_truth (test-only).
"""

from pathlib import Path
from typing import Tuple, Iterable, List
import numpy as np
import pandas as pd
from astropy.io import fits

# Minimal constants
GLOB_PATTERNS = ("*_wcs_aligned.fits", "*_aligned.fits")  # search order
CLIP_SIGMA = 5.0  # for robust_norm


# -------------------- helpers --------------------
def loomed_median(a, b):
    # median of two arrays = average of min/max
    return 0.5 * (np.minimum(a, b) + np.maximum(a, b))

def leave_one_out(F1, F2, F3):
    T1 = loomed_median(F2, F3)
    T2 = loomed_median(F1, F3)
    T3 = loomed_median(F1, F2)
    return F1 - T1, F2 - T2, F3 - T3  # S1, S2, S3

def robust_norm(img, clip_sigma: float = CLIP_SIGMA):
    # Clean NaN/Inf FIRST before any calculations
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med))) + 1e-6
    z = (img - med) / (1.4826 * mad)
    z = np.clip(z, -clip_sigma, clip_sigma)
    return z.astype(np.float32)

def crop_pad(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """
    Crop a size x size patch centered at (cx,cy) (0-indexed).
    Reflect-pad if the window goes out of bounds.
    """
    h, w = img.shape
    r = size // 2
    x0, x1 = int(np.floor(cx)) - r, int(np.floor(cx)) + r
    y0, y1 = int(np.floor(cy)) - r, int(np.floor(cy)) + r

    pad_l = max(0, -x0); pad_r = max(0, x1 - w)
    pad_b = max(0, -y0); pad_t = max(0, y1 - h)
    if pad_l or pad_r or pad_b or pad_t:
        img = np.pad(img, ((pad_b, pad_t), (pad_l, pad_r)), mode="reflect")
        x0 += pad_l; x1 += pad_l; y0 += pad_b; y1 += pad_b

    return img[y0:y1, x0:x1]

def build_channels(F1, F2, F3):
    # 5 channels = F1, F2, F3, S2, S3 (normalize each)
    _S1, S2, S3 = leave_one_out(F1, F2, F3)
    chans = [F1, F2, F3, S2, S3]
    return np.stack([robust_norm(c) for c in chans], axis=0)  # (5,H,W)

def load_frames(folder: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fdir = Path(folder) / "aligned"
    files: List[Path] = []
    for pat in GLOB_PATTERNS:
        files = sorted(fdir.glob(pat))
        if len(files) >= 3:
            break
    if len(files) < 3:
        raise FileNotFoundError(f"Need 3 aligned FITS in {fdir} (patterns: {GLOB_PATTERNS}), found {len(files)}")

    # Clean NaN/Inf immediately after loading
    F1 = np.nan_to_num(fits.getdata(files[0]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    F2 = np.nan_to_num(fits.getdata(files[1]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    F3 = np.nan_to_num(fits.getdata(files[2]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return F1, F2, F3

def ds9_to_zero(xy: Tuple[float, float]) -> Tuple[float, float]:
    # DS9 (1-indexed) -> 0-indexed
    return xy[0] - 1.0, xy[1] - 1.0

def save_npz(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), x=arr)


# -------------------- public API --------------------
def build_test_stamps_for_candidates(
    folder_dir: Path,
    candidates_csv: Path,
    out_root: Path,
    stamp: int = 64
) -> Path:
    """
    Build stamps for all candidate rows belonging to this folder.

    Returns: <out_root>/<folder>
    """
    folder_dir = Path(folder_dir)
    folder = folder_dir.name

    # Load frames and precompute channels once per folder
    F1, F2, F3 = load_frames(folder_dir)
    chans_full = build_channels(F1, F2, F3)  # (5,H,W)

    # Load and filter candidate rows for this folder
    cand = pd.read_csv(candidates_csv)
    cand.columns = [c.strip().lower() for c in cand.columns]
    if "folder" not in cand.columns:
        raise KeyError("candidates_csv must include a 'folder' column")
    cand = cand[cand["folder"].astype(str).str.strip() == str(folder)].copy()
    if cand.empty:
        raise FileNotFoundError(f"No candidates for folder='{folder}' in '{candidates_csv}'")

    out_dir = Path(out_root) / folder / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, r in cand.reset_index(drop=True).iterrows():
        # Prefer F2 for centering; fallback to F1 if missing/NaN
        x = r.get("f2_x", np.nan); y = r.get("f2_y", np.nan)
        if not np.isfinite(x) or not np.isfinite(y):
            x = r.get("f1_x", np.nan); y = r.get("f1_y", np.nan)
        if not np.isfinite(x) or not np.isfinite(y):
            continue  # skip unusable row

        x0, y0 = ds9_to_zero((float(x), float(y)))
        crop = np.stack([crop_pad(ch, x0, y0, stamp) for ch in chans_full], axis=0)  # (5,S,S)

        rel = Path("test") / f"{folder}__cand__{i:07d}.npz"
        save_npz(out_dir / rel.name, crop)

        rows.append({
            "id": int(i), "split": "test",
            "folder": folder, "cx": x0, "cy": y0,
            "label": -1, "source": "cand",
            "stamp_path": str((Path(folder) / rel).as_posix()),
            "f1_x": r.get("f1_x", np.nan), "f1_y": r.get("f1_y", np.nan),
            "f2_x": r.get("f2_x", np.nan), "f2_y": r.get("f2_y", np.nan),
            "f3_x": r.get("f3_x", np.nan), "f3_y": r.get("f3_y", np.nan),
        })

    meta_path = Path(out_root) / folder / "metadata.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    return Path(out_root) / folder


def build_all_test_stamps(
    root_dir: Path,
    candidates_csv: Path,
    out_root: Path,
    stamp: int = 64
) -> List[Path]:
    """
    Build stamps for all distinct folders present in candidates_csv.
    Returns list of folder output roots (<out_root>/<folder>).
    """
    cand = pd.read_csv(candidates_csv)
    cand.columns = [c.strip().lower() for c in cand.columns]
    if "folder" not in cand.columns:
        raise KeyError("candidates_csv must include a 'folder' column")

    built: List[Path] = []
    for folder in sorted(cand["folder"].astype(str).unique()):
        fdir = Path(root_dir) / folder
        try:
            out = build_test_stamps_for_candidates(fdir, candidates_csv, out_root, stamp=stamp)
            built.append(out)
        except Exception as e:
            print(f"[SKIP] {folder}: {e}")
    return built

