from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_exact

# ========== CONFIG ==========
#DATA_ROOT = Path(r"C:\Users\kavya\Documents\cleaned_data\cleaned_data")
DATA_ROOT = Path(r"C:\Users\kavya\Downloads\Copies-20251019T170739Z-1-002\Copies\20240815\A119HJY\phot\out_clear")  # root folder containing subdirs
FITS_EXTS = (".fits", ".fit", ".fz", ".fts")
ALIGN_DIR = "aligned"
WCS_DIR = "wcs_solved"     # ⭐ NEW
# =============================


def is_fits_corrupt(fpath: Path) -> bool:
    """Return True if FITS file is unreadable or empty."""
    try:
        with fits.open(fpath, ignore_missing_end=True) as hdul:
            data = hdul[0].data
            if data is None or data.size == 0:
                return True
    except Exception:
        return True
    return False


def load_fits_with_wcs(fits_path: Path, wcs_path: Path):
    """Return (data, header) with WCS header merged in memory."""
    with fits.open(fits_path) as hdul_fits, fits.open(wcs_path) as hdul_wcs:
        hdr_fits = hdul_fits[0].header.copy()
        hdr_wcs = hdul_wcs[0].header
        for key in hdr_wcs:
            if key not in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "COMMENT", "HISTORY"):
                hdr_fits[key] = hdr_wcs[key]
        return hdul_fits[0].data, hdr_fits


def align_folder(folder: Path) -> bool:
    """Inject WCS headers, save WCS-solved FITS, then align."""
    fits_files = sorted([f for f in folder.glob("*") if f.suffix.lower() in FITS_EXTS])
    if not fits_files:
        print(f"⚠️ No FITS files found in {folder.name}")
        return False

    # Check for corrupt FITS
    for f in fits_files:
        if is_fits_corrupt(f):
            print(f"🚫 Skipping {folder.name}: corrupt FITS ({f.name})")
            return False

    # Verify matching .wcs files exist
    for f in fits_files:
        if not (folder / f"{f.stem}.wcs").exists():
            print(f"⚠️ Missing WCS for {f.name} — skipping folder {folder.name}")
            return False

    # ⭐ NEW: create wcs_solved directory
    wcs_dir = folder / WCS_DIR
    wcs_dir.mkdir(exist_ok=True)

    # Load and save each FITS with merged WCS
    frames = []
    for f in fits_files:
        wcs_file = folder / f"{f.stem}.wcs"
        data, hdr = load_fits_with_wcs(f, wcs_file)

        # ⭐ NEW: save to wcs_solved
        solved_path = wcs_dir / f"{f.stem}_wcs.fits"
        fits.writeto(solved_path, data, hdr, overwrite=True)
        frames.append((solved_path, data, hdr))
        print(f"💾 [WCS-solved] {f.name} → {solved_path.name}")

    # Reference frame = middle one
    ref_data, ref_hdr = frames[len(frames)//2][1], frames[len(frames)//2][2]
    ref_wcs = WCS(ref_hdr)

    out_dir = folder / ALIGN_DIR
    out_dir.mkdir(exist_ok=True)

    print(f"\n📂 Aligning {folder.name} using {frames[len(frames)//2][0].name} as reference")

    # Align each WCS-solved frame
    for f_solved, data, hdr in frames:
        wcs = WCS(hdr)
        reproj, _ = reproject_exact((data, wcs), ref_wcs, shape_out=ref_data.shape)
        out = out_dir / f"{f_solved.stem}_aligned.fits"
        fits.writeto(out, reproj, ref_hdr, overwrite=True)
        print(f"✅ [Aligned] {f_solved.name} → {out.name}")

    print(f"✅ Completed alignment for {folder.name}\n")
    return True


def main():
    #folders = sorted(DATA_ROOT.glob("202*__*"))
    folders = sorted(DATA_ROOT.glob("red"))
    print(f"Found {len(folders)} folders under {DATA_ROOT}\n")

    aligned_count = 0
    skipped_count = 0

    for folder in folders:
        if "comet" in folder.name.lower():
            print(f"🚫 Skipping comet folder: {folder.name}")
            skipped_count += 1
            continue

        try:
            success = align_folder(folder)
            if success:
                aligned_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"⚠️ Error processing {folder.name}: {e}")
            skipped_count += 1

    print("\n================== SUMMARY ==================")
    print(f"✅ Folders successfully aligned: {aligned_count}")
    print(f"🚫 Folders skipped or failed:   {skipped_count}")
    print(f"📂 Total folders processed:     {aligned_count + skipped_count}")
    print("=============================================\n")


if __name__ == "__main__":
    main()

