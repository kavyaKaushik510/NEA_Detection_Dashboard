# --- In-memory wrapper for pipeline_core -------------------------------
from pathlib import Path
import numpy as np
import pandas as pd

# re-use your existing implementations
# (same file or import from append_snr_gmag)
from Core_pipeline.append_snr_gmag import (
    snr_at_xy,
    measure_g_on_three,
    build_calibrator_mookodi,
)
from Core_pipeline.append_snr_gmag import load_image_and_wcs, robust_bg  # if needed

SUBDIR = "aligned"

def _find_aligned_triplet(folder: Path) -> list[Path]:
    d = Path(folder) / SUBDIR
    if not d.exists():
        return []
    files = sorted(d.glob("*_wcs_aligned*.fits"))
    if len(files) < 3:
        files = sorted(d.glob("*_aligned*.fits"))
    return files[:3]

def _normalize_folder_match(series: pd.Series, target: Path) -> pd.Series:
    tgt = Path(target).name.strip().lower()
    def ok(v):
        s = str(v).strip()
        # match if same basename or endswith the folder name
        return (Path(s).name.strip().lower() == tgt) or s.replace("\\","/").rstrip("/").endswith("/"+tgt)
    return series.astype(str).map(ok)

def augment_snr_gmag_df(df: pd.DataFrame, folder: Path) -> pd.DataFrame:
    """Compute snr1/2/3, gmag1/2/3 (+mean/std) for rows of this folder, in-place."""
    # Ensure columns exist
    for c in ["snr1","snr2","snr3","gmag1","gmag2","gmag3","gmag_mean","gmag_std"]:
        if c not in df.columns:
            df[c] = np.nan

    # Resolve aligned FITS
    fits_names = _find_aligned_triplet(folder)
    if len(fits_names) < 3:
        print(f"[augment_snr_gmag_df] WARN: {folder} has <3 aligned FITS under '{SUBDIR}'")
        return df
    f1p, f2p, f3p = fits_names

    # Calibrator once per folder (exact process)
    try:
        calib = build_calibrator_mookodi(f2p)
    except Exception as e:
        print(f"[augment_snr_gmag_df] G-calibration failed: {e}")
        calib = None

    # Rows to update
    if "folder" in df.columns:
        mask = _normalize_folder_match(df["folder"], folder)
        if not mask.any():
            # Fallback: if nothing matched, operate on all rows (safer than no-op)
            print("[augment_snr_gmag_df] NOTE: no 'folder' match; updating ALL rows")
            mask = pd.Series(True, index=df.index)
    else:
        mask = pd.Series(True, index=df.index)

    for i, row in df[mask].iterrows():
        # --- SNR per frame (keep your exact coordinate usage) ---
        try:
            s1 = snr_at_xy(f1p, float(row["f1_x"]), float(row["f1_y"]))
            s2 = snr_at_xy(f2p, float(row["f2_x"]), float(row["f2_y"]))
            s3 = snr_at_xy(f3p, float(row["f3_x"]), float(row["f3_y"]))
            df.at[i, "snr1"], df.at[i, "snr2"], df.at[i, "snr3"] = s1, s2, s3
        except Exception as e:
            print(f"[augment_snr_gmag_df] SNR failed @ row {i}: {e}")

        # --- Gaia G magnitudes (exact process: calibrated; NaN if no calibrator) ---
        try:
            x1, y1 = float(row["f1_x"]), float(row["f1_y"])
            x2, y2 = float(row["f2_x"]), float(row["f2_y"])
            x3, y3 = float(row["f3_x"]), float(row["f3_y"])
            g1, g2, g3 = measure_g_on_three(f1p, f2p, f3p, (x1, y1, x2, y2, x3, y3), calib)
            df.at[i, "gmag1"], df.at[i, "gmag2"], df.at[i, "gmag3"] = g1, g2, g3
            arr = np.array([g1, g2, g3], float)
            if np.isfinite(arr).any():
                df.at[i, "gmag_mean"] = float(np.nanmean(arr))
                df.at[i, "gmag_std"]  = float(np.nanstd(arr))
        except Exception as e:
            print(f"[augment_snr_gmag_df] Gmag failed @ row {i}: {e}")

    # UI/Pipeline compatibility aliases (so Streamlit columns aren’t blank)
    # f1_snr/f2_snr/f3_snr and single gmag (mean)
    df.loc[mask, "f1_snr"] = df.loc[mask, "snr1"]
    df.loc[mask, "f2_snr"] = df.loc[mask, "snr2"]
    df.loc[mask, "f3_snr"] = df.loc[mask, "snr3"]
    df.loc[mask, "gmag"]   = df.loc[mask, "gmag_mean"]

    # --- display rounding, keep raw columns untouched for ML ---
    for c in ["gmag1", "gmag2", "gmag3", "gmag_mean"]:
        if c in df.columns:
            df[c + "_disp"] = df[c].round(1)


    return df
