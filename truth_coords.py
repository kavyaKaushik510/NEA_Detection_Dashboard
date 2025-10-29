from pathlib import Path
import re
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

def parse_mpc_to_ds9(folder, aligned_files):
    """Parse first MPC file to DS9 pixel coords using WCS of each aligned frame."""
    mpc_files = list(Path(folder).glob("*MPC*txt"))
    if not mpc_files:
        print("⚠️ No MPC file found — skipping")
        return None
    mpc = mpc_files[0]

    coords = []
    with open(mpc) as f:
        for line in f:
            m = re.search(r"(\d{2}\s+\d{2}\s+\d{2}(?:\.\d+)?)\s+([+-]\d{2}\s+\d{2}\s+\d{2}(?:\.\d+)?)", line)
            if m:
                ra  = m.group(1).replace(" ","h",1).replace(" ","m",1)+"s"
                dec = m.group(2).replace(" ","d",1).replace(" ","m",1)+"s"
                coords.append(SkyCoord(ra,dec,frame="icrs"))

    n = len(coords)
    if n < 2:
        print(f"⚠️ Only {n} MPC coords in {folder}")
        return None

    pixel_positions = []
    for c, f in zip(coords[:3], aligned_files[:3]):  # handle 2 or 3
        hdr = fits.getheader(f)
        wcs = WCS(hdr)
        pixel_positions.append(wcs.world_to_pixel(c))

    # pad to 3 for consistency
    while len(pixel_positions) < 3:
        pixel_positions.append((np.nan, np.nan))
    return pixel_positions