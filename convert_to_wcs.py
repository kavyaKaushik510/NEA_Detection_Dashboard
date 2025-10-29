# convert_to_wcs.py
# ---------------------------------------------------------------------------
# Compute RA/Dec for Top-3 scored tracks and write a compact CSV.
# Default sort is by 'score_hybrid' (falls back if missing).
# ---------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

# ----------------------------- Helpers ---------------------------------------

def _aligned_triplet_from_folder(folder: Path) -> List[Path]:
    aligned_dir = Path(folder) / "aligned"
    paths = sorted(aligned_dir.glob("*_wcs_aligned.fits"))
    return paths[:3]

def _aligned_triplet_from_row(row: pd.Series) -> Optional[List[Path]]:
    # Optional: allow rows to carry per-frame FITS paths via columns like f1/f2/f3 or f1_path/f2_path/f3_path
    for key in ("f1", "f1_path", "fits_f1"):
        if key in row and pd.notna(row[key]):
            try:
                f1 = Path(str(row[key]))
                base = key.split("1")[0]  # e.g., "f" -> f2/f3; or "f1_path" -> "f_path2"/"f_path3"
                f2_key, f3_key = base + "2", base + "3"
                f2 = Path(str(row.get(f2_key, "")))
                f3 = Path(str(row.get(f3_key, "")))
                if all(p and (Path(p).suffix.lower() in (".fits", ".fit", ".fts", ".fz")) for p in (f1, f2, f3)):
                    return [Path(f1), Path(f2), Path(f3)]
            except Exception:
                pass
    return None

def _xy_to_radec(fits_path: Path, x: float, y: float) -> Tuple[float, float]:
    try:
        with fits.open(fits_path) as hdul:
            w = WCS(hdul[0].header)
            sky = w.pixel_to_world(x, y)  # expects pixel coords consistent with how f*_x/f*_y were measured
            return float(sky.ra.deg), float(sky.dec.deg)
    except Exception:
        return (float("nan"), float("nan"))

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _pick_score_col(df: pd.DataFrame, preferred: str = "score_hybrid") -> Optional[str]:
    # Choose a sensible score column
    if preferred in df.columns:
        return preferred
    fallbacks = ["score_optimized", "score", "cnn_prob", "gb_confidence"]
    return _pick_col(df, fallbacks)

# ----------------------------- Public API ------------------------------------

def attach_top3_radec(
    df_scored: pd.DataFrame,
    folder: Path,
    keep_extra_cols: Optional[List[str]] = None,
    score_col: str = "score_hybrid",
) -> pd.DataFrame:
    """
    Returns a Top-3 dataframe with RA/Dec per frame and useful photometry columns.
    Sorts by score_col (default 'score_hybrid'), with graceful fallback if missing.
    """
    if df_scored is None or df_scored.empty:
        return pd.DataFrame()

    # Pick score column to sort by
    sort_col = _pick_score_col(df_scored, preferred=score_col)
    if sort_col is None:
        # No recognizable score column; just return first 3 rows with NaN RA/Dec
        top3 = df_scored.head(3).copy()
    else:
        top3 = df_scored.sort_values(sort_col, ascending=False).head(3).copy()

    top3 = top3.reset_index(drop=True)
    top3["rank"] = np.arange(1, len(top3) + 1)

    # Resolve aligned FITS (row-specified triplet > folder triplet)
    fpaths = _aligned_triplet_from_row(top3.iloc[0]) or _aligned_triplet_from_folder(folder)
    if len(fpaths) < 3:
        # Fill with NaNs if we can't resolve the triplet
        for c in ["ra1","dec1","ra2","dec2","ra3","dec3"]:
            top3[c] = np.nan
    else:
        f1, f2, f3 = fpaths
        ra1_list, dec1_list, ra2_list, dec2_list, ra3_list, dec3_list = ([] for _ in range(6))
        for _, r in top3.iterrows():
            x1, y1 = float(r.get("f1_x", np.nan)), float(r.get("f1_y", np.nan))
            x2, y2 = float(r.get("f2_x", np.nan)), float(r.get("f2_y", np.nan))
            x3, y3 = float(r.get("f3_x", np.nan)), float(r.get("f3_y", np.nan))

            ra1, dec1 = _xy_to_radec(f1, x1, y1) if np.isfinite(x1) and np.isfinite(y1) else (np.nan, np.nan)
            ra2, dec2 = _xy_to_radec(f2, x2, y2) if np.isfinite(x2) and np.isfinite(y2) else (np.nan, np.nan)
            ra3, dec3 = _xy_to_radec(f3, x3, y3) if np.isfinite(x3) and np.isfinite(y3) else (np.nan, np.nan)

            ra1_list.append(ra1); dec1_list.append(dec1)
            ra2_list.append(ra2); dec2_list.append(dec2)
            ra3_list.append(ra3); dec3_list.append(dec3)

        top3["ra1"], top3["dec1"] = ra1_list, dec1_list
        top3["ra2"], top3["dec2"] = ra2_list, dec2_list
        top3["ra3"], top3["dec3"] = ra3_list, dec3_list

    # Normalize gmag & SNR column names (keep if present)
    g1 = _pick_col(top3, ["gmag1","gmag_f1","gmag_f01","f1_gmag"])
    g2 = _pick_col(top3, ["gmag2","gmag_f2","gmag_f02","f2_gmag"])
    g3 = _pick_col(top3, ["gmag3","gmag_f3","gmag_f03","f3_gmag"])
    if g1 and g1 != "gmag1": top3 = top3.rename(columns={g1:"gmag1"})
    if g2 and g2 != "gmag2": top3 = top3.rename(columns={g2:"gmag2"})
    if g3 and g3 != "gmag3": top3 = top3.rename(columns={g3:"gmag3"})

    s1 = _pick_col(top3, ["snr1","snr_f1","snr_f01","f1_snr","SNR1","SNR_f1"])
    s2 = _pick_col(top3, ["snr2","snr_f2","snr_f02","f2_snr","SNR2","SNR_f2"])
    s3 = _pick_col(top3, ["snr3","snr_f3","snr_f03","f3_snr","SNR3","SNR_f3"])
    if s1 and s1 != "snr1": top3 = top3.rename(columns={s1:"snr1"})
    if s2 and s2 != "snr2": top3 = top3.rename(columns={s2:"snr2"})
    if s3 and s3 != "snr3": top3 = top3.rename(columns={s3:"snr3"})

    # Optional summary stats (include if present)
    gmean = _pick_col(top3, ["gmag_mean_obs","gmag_mean"])
    smean = _pick_col(top3, ["snr_mean_obs","snr_mean"])
    gstd  = _pick_col(top3, ["gmag_std"])
    sstd  = _pick_col(top3, ["snr_std"])

    # Column order for export (score column is dynamic)
    base_cols = [
        "rank", "folder",
    ]
    if sort_col is not None and sort_col in top3.columns:
        base_cols.append(sort_col)  # e.g., 'score_hybrid'

    base_cols += [
        "f1_x","f1_y","ra1","dec1","gmag1","snr1",
        "f2_x","f2_y","ra2","dec2","gmag2","snr2",
        "f3_x","f3_y","ra3","dec3","gmag3","snr3",
    ]
    base_cols = [c for c in base_cols if c in top3.columns]

    # Add common modern scoring columns if present
    for c in ["cnn_prob", "gb_confidence", "score_hybrid"]:
        if c in top3.columns and c not in base_cols:
            base_cols.append(c)

    # Add means/stds if present
    for c in [gmean, gstd, smean, sstd]:
        if c and c in top3.columns and c not in base_cols:
            base_cols.append(c)

    if keep_extra_cols:
        for c in keep_extra_cols:
            if c in top3.columns and c not in base_cols:
                base_cols.append(c)

    return top3[base_cols]

def export_top3_radec_csv(
    df_scored: pd.DataFrame,
    folder: Path,
    out_path: Optional[Path] = None,
    keep_extra_cols: Optional[List[str]] = None,
    score_col: str = "score_hybrid",
) -> Path:
    top3 = attach_top3_radec(df_scored, folder, keep_extra_cols=keep_extra_cols, score_col=score_col)
    if out_path is None:
        out_path = Path(folder) / "top3_tracks_radec.csv"
    top3.to_csv(out_path, index=False)
    return out_path
