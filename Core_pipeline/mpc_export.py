# mpc_export.py
# -----------------------------------------------------------------------------
# Build a classic MPC-style text for one 3-detection track.
# Fixes date formatting, per-frame timestamps, and magnitude fallbacks.
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
import math


# ------------------------- FITS + WCS helpers --------------------------------

def _aligned_triplet(folder: Path) -> List[Path]:
    """Find aligned/*_wcs_aligned.fits as F1,F2,F3 by sorted name."""
    aligned = Path(folder) / "aligned"
    files = sorted(aligned.glob("*_wcs_aligned.fits"))
    return files[:3]


def _xy_to_radec(fits_path: Path, x: float, y: float) -> Tuple[float, float]:
    """Project a single (x,y) to (RA,Dec) in degrees using FITS WCS."""
    with fits.open(fits_path) as hdul:
        w = WCS(hdul[0].header)
        sky = w.pixel_to_world(x, y)
        return float(sky.ra.deg), float(sky.dec.deg)


def _get_time_utc(fits_path: Path) -> Time:
    """
    Robustly read observation time for one frame, returning an astropy Time (UTC).
    Priority:
      1) MJD-OBS (mid-exposure) if present
      2) DATE-AVG (ISO)
      3) DATE-OBS + EXPTIME/2 (mid-exposure)
      4) DATE-OBS
    """
    with fits.open(fits_path) as hdul:
        h = hdul[0].header

        # 1) MJD-OBS
        mjd = h.get("MJD-OBS")
        if mjd is not None:
            try:
                return Time(float(mjd), format="mjd", scale="utc")
            except Exception:
                pass

        # 2) DATE-AVG
        date_avg = h.get("DATE-AVG") or h.get("DATEAVG")
        if date_avg:
            try:
                return Time(date_avg, format="isot", scale="utc")
            except Exception:
                try:
                    return Time(date_avg, format="iso", scale="utc")
                except Exception:
                    pass

        # 3) DATE-OBS (+ EXPTIME/2 if available)
        date_obs = h.get("DATE-OBS") or h.get("DATEOBS")
        if date_obs:
            try:
                t = Time(date_obs, format="isot", scale="utc")
            except Exception:
                # fall back to 'iso' if 'isot' fails
                try:
                    t = Time(date_obs, format="iso", scale="utc")
                except Exception:
                    t = None
            if t is not None:
                exptime = h.get("EXPTIME") or h.get("EXPOSURE") or h.get("ITIME")
                try:
                    if exptime is not None and float(exptime) > 0:
                        return t + float(exptime) / 2.0 / 86400.0  # mid-exposure
                except Exception:
                    pass
                return t

    # Fallback: clearly invalid but defined
    return Time(0.0, format="mjd", scale="utc")


# --------------------------- Formatting helpers ------------------------------

def _hms_from_deg(ra_deg: float) -> Tuple[int, int, float]:
    """RA degrees -> (HH, MM, SS.ss)."""
    total_hours = ra_deg / 15.0
    H = int(total_hours)
    m_float = (total_hours - H) * 60.0
    M = int(m_float)
    S = (m_float - M) * 60.0
    return H, M, S


def _dms_from_deg(dec_deg: float) -> Tuple[str, int, int, float]:
    """Dec degrees -> (sign, DD, MM, SS.s)."""
    sign = "+" if dec_deg >= 0 else "-"
    a = abs(dec_deg)
    D = int(a)
    m_float = (a - D) * 60.0
    M = int(m_float)
    S = (m_float - M) * 60.0
    return sign, D, M, S


def _mpc_date_string(t: Time) -> str:
    """
    MPC classic date: 'YYYY MM DD.ddddd' (5 decimals).
    Compute fractional day from the UTC time-of-day.
    """
    t_utc = t.utc
    y = t_utc.datetime.year
    m = t_utc.datetime.month
    d = t_utc.datetime.day
    sec_of_day = (
        t_utc.datetime.hour * 3600
        + t_utc.datetime.minute * 60
        + t_utc.datetime.second
        + t_utc.datetime.microsecond / 1e6
    )
    frac = sec_of_day / 86400.0  # in [0,1)
    # format WITHOUT an extra "0."—just 'DD.' + 5 decimals
    return f"{y:04d} {m:02d} {d:02d}.{int(round(frac * 1e5)):05d}"


def _fmt_ra(ra_deg: float) -> str:
    """'HH MM SS.ss' with 2 decimals on seconds."""
    H, M, S = _hms_from_deg(ra_deg)
    return f"{H:02d} {M:02d} {S:05.2f}"


def _fmt_dec(dec_deg: float) -> str:
    """'±DD MM SS.s' with 1 decimal on arcsec."""
    sign, D, M, S = _dms_from_deg(dec_deg)
    return f"{sign}{D:02d} {M:02d} {S:04.1f}"


def _mag_str(mag: Optional[float]) -> str:
    """Format magnitude field to one decimal; blank if NaN."""
    if mag is None or not np.isfinite(mag):
        return "     "  # keep spacing if unknown
    return f"{mag:4.1f}"


# ------------------------------ Public API -----------------------------------

MPC_HEADER_DEFAULT = [
    "COD M28",
    "CON N. Erasmus, 1 Observatory Road, Cape Town, South Africa",
    "OBS N. Erasmus, S. Potter, C. van Gend, S. Chandra ",
    "MEA N. Erasmus, T. Ngwane",
    "TEL 1.0-m f/8 Reflector + CCD",
    "NET Gaia-DR2",
    "AC2 nerasmus@saao.ac.za, thobekilesn@gmail.com",
    "ACK NEOCP",
]


def build_mpc_for_track(
    folder: Path,
    row: pd.Series,
    designation: str = "A10Wcxt",
    band: str = "G",
    obs_code: str = "M28",
    header_lines: Optional[List[str]] = None,
) -> str:
    """
    Build an MPC text for the 3 detections of a single scored track.
    Uses aligned FITS for timing (prefer mid-exposure) and WCS coordinates.
    """
    fits_paths = _aligned_triplet(folder)
    if len(fits_paths) < 3:
        raise RuntimeError("Aligned triplet not found under folder/aligned/*.fits")

    # (x,y) from the row
    x1, y1 = float(row.get("f1_x")), float(row.get("f1_y"))
    x2, y2 = float(row.get("f2_x")), float(row.get("f2_y"))
    x3, y3 = float(row.get("f3_x")), float(row.get("f3_y"))

    # RA/Dec per frame
    ra1, dec1 = _xy_to_radec(fits_paths[0], x1, y1)
    ra2, dec2 = _xy_to_radec(fits_paths[1], x2, y2)
    ra3, dec3 = _xy_to_radec(fits_paths[2], x3, y3)

    # Per-frame UTC time (distinct for each)
    t1 = _get_time_utc(fits_paths[0])
    t2 = _get_time_utc(fits_paths[1])
    t3 = _get_time_utc(fits_paths[2])

    # Magnitudes: prefer per-frame; fall back to mean if needed
    g1 = row.get("gmag1", row.get("gmag_f1", np.nan))
    g2 = row.get("gmag2", row.get("gmag_f2", np.nan))
    g3 = row.get("gmag3", row.get("gmag_f3", np.nan))
    if not np.isfinite(g1) or not np.isfinite(g2) or not np.isfinite(g3):
        gmean = row.get("gmag_mean_obs", np.nan)
        if np.isfinite(gmean):
            g1 = g1 if np.isfinite(g1) else float(f"{gmean:.1f}")
            g2 = g2 if np.isfinite(g2) else float(f"{gmean:.1f}")
            g3 = g3 if np.isfinite(g3) else float(f"{gmean:.1f}")

    # Header
    header = header_lines if header_lines else MPC_HEADER_DEFAULT

    # Body lines (mirror your sample spacing)
    def body_line(t: Time, ra: float, dec: float, mag: Optional[float]) -> str:
        date_s = _mpc_date_string(t)         # e.g., '2023 08 07.83523'
        ra_s   = _fmt_ra(ra)                 # 'HH MM SS.ss'
        dec_s  = _fmt_dec(dec)               # '±DD MM SS.s'
        mag_s  = _mag_str(mag)               # '18.5' or blanks
        # Two leading spaces, designation (left-justified to 7),
        # two spaces, 'C', date, space, RA, space, Dec,
        # 12 spaces, mag, space, band, 6 spaces, station code.
        return f"  {designation:<7s}  C{date_s} {ra_s} {dec_s}            {mag_s} {band:<1}      {obs_code}"

    lines = [
        *header,
        "",
        body_line(t1, ra1, dec1, g1),
        body_line(t2, ra2, dec2, g2),
        body_line(t3, ra3, dec3, g3),
        ""
    ]
    return "\n".join(lines)
