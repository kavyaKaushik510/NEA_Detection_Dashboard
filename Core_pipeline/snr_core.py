# snr_core.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from astropy.io import fits

# ---- Global defaults (match everywhere) ----
R_SIGNAL_DEFAULT = 3
R_BG1_DEFAULT    = 8
R_BG2_DEFAULT    = 15
CLIP_BG_SIGMA_DEFAULT = 5.0
USE_FLOAT64 = True   # force precision

def _as_float(a):
    return np.asarray(a, np.float64 if USE_FLOAT64 else np.float32)

def mad_std(a: np.ndarray) -> float:
    a = _as_float(a)
    a = a[np.isfinite(a)]
    if a.size == 0: return np.nan
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return 1.4826 * mad

def read_first_image_hdu(fpath: Path) -> tuple[np.ndarray, dict]:
    with fits.open(str(fpath)) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "data") and isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 2:
                data = _as_float(hdu.data)
                header = dict(hdu.header)
                return data, header
    raise ValueError(f"No 2D image HDU in {fpath}")

def snr_at_xy(
    fits_path: Path,
    x: float,
    y: float,
    r_signal: int = R_SIGNAL_DEFAULT,
    r_bg1: int = R_BG1_DEFAULT,
    r_bg2: int = R_BG2_DEFAULT,
    clip_bg_sigma: float = CLIP_BG_SIGMA_DEFAULT,
) -> float:
    """Robust local SNR with sigma-clipped annulus, median + MAD σ."""
    data, _ = read_first_image_hdu(fits_path)
    h, w = data.shape
    # IMPORTANT: use *float* pixel centers, do not round
    if not (0 <= x < w and 0 <= y < h): return np.nan

    yy, xx = np.ogrid[:h, :w]
    r2 = (xx - x)**2 + (yy - y)**2

    ap_mask = r2 <= (r_signal**2)
    bg_mask = (r2 >= (r_bg1**2)) & (r2 <= (r_bg2**2))

    ap_vals = data[ap_mask]
    bg_vals = data[bg_mask]

    # robust annulus stats
    bg_med = np.median(bg_vals[np.isfinite(bg_vals)])
    bg_sig = mad_std(bg_vals)

    if np.isfinite(bg_sig) and clip_bg_sigma and clip_bg_sigma > 0:
        keep = np.abs(bg_vals - bg_med) <= (clip_bg_sigma * bg_sig)
        bg_vals = bg_vals[keep]
        bg_med = np.median(bg_vals[np.isfinite(bg_vals)])
        bg_sig = mad_std(bg_vals)

    n_ap = int(np.count_nonzero(ap_mask))
    if n_ap == 0 or not np.isfinite(bg_sig): return np.nan

    net_flux = float(np.nansum(ap_vals) - bg_med * n_ap)
    denom    = float(bg_sig * np.sqrt(n_ap) + 1e-6)
    return float(net_flux / denom)

def snr_triplet(
    f1: Path, f2: Path, f3: Path,
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float,
    r_signal: int = R_SIGNAL_DEFAULT,
    r_bg1: int = R_BG1_DEFAULT,
    r_bg2: int = R_BG2_DEFAULT,
    clip_bg_sigma: float = CLIP_BG_SIGMA_DEFAULT,
) -> tuple[float, float, float]:
    s1 = snr_at_xy(f1, x1, y1, r_signal, r_bg1, r_bg2, clip_bg_sigma) if f1 else np.nan
    s2 = snr_at_xy(f2, x2, y2, r_signal, r_bg1, r_bg2, clip_bg_sigma) if f2 else np.nan
    s3 = snr_at_xy(f3, x3, y3, r_signal, r_bg1, r_bg2, clip_bg_sigma) if f3 else np.nan
    return (s1, s2, s3)
