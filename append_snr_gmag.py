# -*- coding: utf-8 -*-
"""
Combine SNR + Gmag augmentation into one pass.

Reads candidate CSV (from motion linker output),
computes per-frame SNR and G-band magnitudes,
and overwrites the same CSV with new columns.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astroquery.gaia import Gaia
import sep
from snr_core import snr_at_xy

# === CONFIG ===
ROOT_DIR = Path(r"C:\Users\kavya\Documents\valid_copies")
CANDIDATES_CSV = Path(r"C:\Users\kavya\Documents\pipeline_test\all_candidates_copies.csv")
SUBDIR = "aligned"
CANDIDATES_OUT = CANDIDATES_CSV.with_name(CANDIDATES_CSV.stem + "_snr_gmag_new.csv")

# --- SNR constants ---
R_SIGNAL = 3
R_BG1 = 8
R_BG2 = 15
CLIP_BG_SIGMA = 5.0

# --- Photometry / calibration constants ---
APER_R_PX = 4.0
THRESH_SIGMA = 1.5
DEBLEND_CONT = 0.005
MATCH_RADIUS_PX = 3.0
PEAK_MIN, PEAK_MAX = 2000.0, 40000.0
GAIA_TABLE = "gaiadr2.gaia_source"
BOX_DEG = 0.166667  # ~10 arcmin box

# -----------------------------------------------------
# Helper functions (common to both SNR + gmag parts)
# -----------------------------------------------------
def _read_fits_first_image_hdu(fpath: Path):
    with fits.open(fpath) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "data") and isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 2:
                return np.asarray(hdu.data, float), dict(hdu.header)
    raise ValueError(f"No 2D image found in {fpath}")

def _mad_std(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return 1.4826 * mad

def robust_bg(img):
    b = sep.Background(img)
    return b, img - b.back(), b.globalrms

def load_image_and_wcs(path: Path):
    with fits.open(path) as hdul:
        data = np.ascontiguousarray(np.nan_to_num(hdul[0].data, copy=False).astype(np.float64))
        hdr = hdul[0].header
    w = WCS(hdr)
    return data, hdr, w

# --- NEW: exposure time + time-sort helpers (minimal) ------------------------
def _get_exptime(hdr) -> float:
    for key in ("EXPTIME", "EXPOSURE", "TEXPTIME"):
        if key in hdr and np.isfinite(hdr[key]):
            try:
                val = float(hdr[key])
                if val > 0:
                    return val
            except Exception:
                pass
    return 1.0  # safe fallback if header lacks exposure

def _get_mjd(hdr) -> float:
    if "MJD-OBS" in hdr:
        try:
            return float(hdr["MJD-OBS"])
        except Exception:
            pass
    if "DATE-OBS" in hdr:
        try:
            return Time(hdr["DATE-OBS"]).mjd
        except Exception:
            pass
    return np.nan

def _time_sort_triplet(paths):
    items = []
    for p in paths:
        try:
            hdr = fits.getheader(p, 0)
            mjd = _get_mjd(hdr)
        except Exception:
            mjd = np.nan
        items.append((mjd, p))
    # NaNs sink to end but order is still deterministic
    items.sort(key=lambda t: (np.isnan(t[0]), t[0]))
    return [p for _, p in items]

# -----------------------------------------------------
# Gmag calibration utilities
# -----------------------------------------------------
def detect_and_photometer_mookodi(fits_path: Path):
    data, hdr, w = load_image_and_wcs(fits_path)
    b, sub, rms = robust_bg(data)
    minarea = max(5, int(np.pi * APER_R_PX**2 / 10))
    objs = sep.extract(sub, thresh=THRESH_SIGMA, err=rms, minarea=minarea, deblend_cont=DEBLEND_CONT)
    if objs is None or len(objs) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), hdr, w
    flux, fluxerr, flags = sep.sum_circle(sub, objs["x"], objs["y"], APER_R_PX, err=rms)
    peak = objs["peak"] if "peak" in objs.dtype.names else np.zeros_like(flux)
    keep = (flux > 0) & np.isfinite(flux) & (peak >= PEAK_MIN) & (peak <= PEAK_MAX)
    return objs["x"][keep], objs["y"][keep], flux[keep], peak[keep], hdr, w

def build_calibrator_mookodi(fits_path: Path):
    """Linear fit between instrumental and Gaia G magnitudes."""
    x, y, flux, peak, hdr, w = detect_and_photometer_mookodi(fits_path)
    if len(flux) == 0:
        return None

    # FIX: normalise by exposure time for a universal calibrator
    exptime = _get_exptime(hdr)
    m_inst = -2.5 * np.log10(np.clip(flux / exptime, 1e-12, None))

    ra_c, dec_c = w.wcs.crval
    q = f"""
        SELECT ra, dec, phot_g_mean_mag
        FROM {GAIA_TABLE}
        WHERE CONTAINS(
          POINT('ICRS', ra, dec),
          BOX('ICRS', {ra_c}, {dec_c}, {BOX_DEG}, {BOX_DEG})
        )=1
    """
    stars = Gaia.launch_job_async(q).get_results()
    if len(stars) == 0:
        return None

    gx, gy = w.wcs_world2pix(np.array(stars["ra"]), np.array(stars["dec"]), 0)
    det = np.column_stack([x, y])
    cat = np.column_stack([gx, gy])
    d2 = ((det[:, None, 0] - cat[None, :, 0])**2 + (det[:, None, 1] - cat[None, :, 1])**2)
    nn_idx = np.argmin(d2, axis=1)
    nn_dist_px = np.sqrt(d2[np.arange(len(det)), nn_idx])
    use = nn_dist_px < MATCH_RADIUS_PX
    if not np.any(use):
        return None

    mi = m_inst[use]
    G = np.array(stars["phot_g_mean_mag"])[nn_idx[use]]
    b, a = np.polyfit(mi, G, 1)
    return {"a": float(a), "b": float(b)}

def measure_g_on_three(f1_path, f2_path, f3_path, coords, calib):
    if calib is None:
        return (np.nan, np.nan, np.nan)

    def m_at(path, x, y):
        data, hdr, w = load_image_and_wcs(path)
        b, sub, rms = robust_bg(data)
        flux, fluxerr, flags = sep.sum_circle(sub, np.array([x]), np.array([y]), APER_R_PX, err=rms)
        f = float(flux[0])
        if not np.isfinite(f) or f <= 0:
            return np.nan
        # FIX: use counts-per-second for consistency across frames
        exptime = _get_exptime(hdr)
        m_inst = -2.5 * np.log10(max(f / max(exptime, 1e-9), 1e-12))
        return calib["a"] + calib["b"] * m_inst

    x1, y1, x2, y2, x3, y3 = coords
    g1 = m_at(f1_path, x1, y1)
    g2 = m_at(f2_path, x2, y2)
    g3 = m_at(f3_path, x3, y3)
    return g1, g2, g3

# -----------------------------------------------------
# Main processing
# -----------------------------------------------------
def main():
    df = pd.read_csv(CANDIDATES_CSV)
    # df = df.head(100)  # DEBUG LIMIT
    print(f"🔍 Loaded {len(df)} candidates")

    # ✅ Preserve all original columns
    for col in ["snr1", "snr2", "snr3", "gmag1", "gmag2", "gmag3", "gmag_mean", "gmag_std"]:
        if col not in df.columns:
            df[col] = np.nan

    for folder_name, dfg in df.groupby("folder", sort=False):
        folder_dir = ROOT_DIR / folder_name
        fits_names = sorted((folder_dir / SUBDIR).glob("*_wcs_aligned*.fits"))
        if len(fits_names) < 3:
            print(f"[WARN] {folder_name}: less than 3 aligned FITS")
            continue

        # FIX: time-sort to map correctly to f1/f2/f3 coords
        f1p, f2p, f3p = _time_sort_triplet(fits_names)[:3]

        # --- Gmag calibration once per folder (on mid frame) ---
        try:
            calib = build_calibrator_mookodi(f2p)
        except Exception as e:
            print(f"⚠️ {folder_name}: calibration failed ({e})")
            calib = None

        for i in dfg.index:
            row = df.loc[i]
            coords = (row["f1_x"], row["f1_y"], row["f2_x"], row["f2_y"], row["f3_x"], row["f3_y"])

            # --- SNR ---
            try:
                s1 = snr_at_xy(f1p, float(row["f1_x"]), float(row["f1_y"]))
                s2 = snr_at_xy(f2p, float(row["f2_x"]), float(row["f2_y"]))
                s3 = snr_at_xy(f3p, float(row["f3_x"]), float(row["f3_y"]))
            except Exception as e:
                print(f"[WARN] SNR failed for {folder_name} row {i}: {e}")
                s1 = s2 = s3 = np.nan

            df.at[i, "snr1"] = s1
            df.at[i, "snr2"] = s2
            df.at[i, "snr3"] = s3

            # --- Gmag ---
            try:
                g1, g2, g3 = measure_g_on_three(f1p, f2p, f3p, coords, calib)
                arr = np.array([g1, g2, g3], float)
                df.at[i, "gmag1"] = g1
                df.at[i, "gmag2"] = g2
                df.at[i, "gmag3"] = g3
                if np.isfinite(arr).any():
                    df.at[i, "gmag_mean"] = np.nanmean(arr)
                    df.at[i, "gmag_std"] = np.nanstd(arr)
            except Exception as e:
                print(f"[WARN] Gmag failed for {folder_name} row {i}: {e}")

        print(f"✅ {folder_name}: done ({len(dfg)} candidates)")

    df.to_csv(CANDIDATES_OUT, index=False)
    print(f"💾 Wrote {CANDIDATES_OUT} with {len(df)} rows and {len(df.columns)} columns")

def augment_snr_gmag_df(df: pd.DataFrame, folder: str | Path = None) -> pd.DataFrame:
    """
    Dashboard wrapper: apply the SAME logic as the script's main() but
    operate on an in-memory DataFrame and return it (no CSV I/O).
    Uses your exact process: lexicographic sort for *_wcs_aligned*.fits[:3],
    build_calibrator_mookodi() on F2, measure_g_on_three() per row, snr_at_xy().
    """
    out = df.copy()

    # Ensure required columns exist
    for col in ["snr1", "snr2", "snr3", "gmag1", "gmag2", "gmag3", "gmag_mean", "gmag_std"]:
        if col not in out.columns:
            out[col] = np.nan

    # Restrict to one folder if provided
    if folder is not None:
        group_iter = [(folder, out[out["folder"] == folder])]
    else:
        group_iter = out.groupby("folder", sort=False)

    # Per-folder processing (same as your script)
    for folder_name, dfg in group_iter:
        if len(dfg) == 0:
            continue
        folder_dir = ROOT_DIR / str(folder_name)
        fits_names = sorted((folder_dir / SUBDIR).glob("*_wcs_aligned*.fits"))
        if len(fits_names) < 3:
            # keep NaNs if fewer than 3 frames
            continue

        f1p, f2p, f3p = fits_names[:3]

        # Calibrate once per folder using your function
        try:
            calib = build_calibrator_mookodi(f2p)
        except Exception:
            calib = None

        # Row-wise SNR + Gmag (your exact per-row flow)
        for i in dfg.index:
            row = out.loc[i]
            coords = (
                row["f1_x"], row["f1_y"],
                row["f2_x"], row["f2_y"],
                row["f3_x"], row["f3_y"]
            )

            # SNR
            try:
                s1 = snr_at_xy(f1p, float(row["f1_x"]), float(row["f1_y"]))
                s2 = snr_at_xy(f2p, float(row["f2_x"]), float(row["f2_y"]))
                s3 = snr_at_xy(f3p, float(row["f3_x"]), float(row["f3_y"]))
                out.at[i, "snr1"] = s1
                out.at[i, "snr2"] = s2
                out.at[i, "snr3"] = s3
            except Exception:
                # leave as NaN if SNR calc fails
                pass

            # Gmag
            try:
                g1, g2, g3 = measure_g_on_three(f1p, f2p, f3p, coords, calib)
                arr = np.array([g1, g2, g3], float)
                out.at[i, "gmag1"] = g1
                out.at[i, "gmag2"] = g2
                out.at[i, "gmag3"] = g3
                if np.isfinite(arr).any():
                    out.at[i, "gmag_mean"] = np.nanmean(arr)
                    out.at[i, "gmag_std"]  = np.nanstd(arr)
            except Exception:
                # leave as NaN if photometry fails
                pass

    return out
