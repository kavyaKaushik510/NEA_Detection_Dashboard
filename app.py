# app.py
import streamlit as st
from pathlib import Path
import tempfile, shutil
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
import pandas as pd
from Core_pipeline.convert_to_wcs import export_top3_radec_csv
from Core_pipeline.mpc_export import build_mpc_for_track 

# Your modules
from pipeline_core import run_full_pipeline
from Core_pipeline.alignment import align_folder

st.set_page_config(page_title="Asteroid Detection (Confidence Score)", layout="wide")
st.title("🌌 NEA Detection for Lesedi 1-m Telescope at SAAO")

# -------------------------------
# Small helpers
# -------------------------------
def unzip_to_tmp(uploaded_file) -> Path:
    tmpdir = Path(tempfile.mkdtemp())
    zpath = tmpdir / "input.zip"
    with open(zpath, "wb") as f:
        f.write(uploaded_file.read())
    shutil.unpack_archive(str(zpath), str(tmpdir))
    candidates = [p for p in tmpdir.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if not candidates:
        root = tmpdir / "unzipped"
        root.mkdir(exist_ok=True)
        for p in tmpdir.iterdir():
            if p.is_file() and p.name != "input.zip":
                shutil.move(str(p), root / p.name)
        candidates = [root]
    return candidates[0] if len(candidates) == 1 else tmpdir

def have_aligned_triplet(folder: Path) -> bool:
    aligned_dir = folder / "aligned"
    if not aligned_dir.exists():
        return False
    files = sorted(aligned_dir.glob("*_wcs_aligned.fits"))
    return len(files) >= 3

def have_raw_fits_plus_wcs(folder: Path) -> bool:
    fits_files = [f for f in folder.glob("*") if f.suffix.lower() in (".fits", ".fit", ".fz", ".fts")]
    if len(fits_files) < 3:
        return False
    return all((folder / f"{f.stem}.wcs").exists() for f in fits_files)

# ---- Small triplet renderer (thin, light red circle + (x,y)) ----
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
import streamlit as st

def show_all_three_aligned_frames(folder: Path, top3: pd.DataFrame):
    """
    Display F1, F2, F3 aligned frames side-by-side,
    showing only the top-3 tracks (light red lines + circles).
    """
    aligned_dir = folder / "aligned"
    aligned = sorted(aligned_dir.glob("*_wcs_aligned.fits"))
    if len(aligned) < 3:
        st.warning("No aligned FITS found for visualisation.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, ax in enumerate(axes):
        with fits.open(aligned[i]) as h:
            img = h[0].data.astype(float)
        norm = simple_norm(img, "asinh", percent=99)
        ax.imshow(img, cmap="gray", origin="lower", norm=norm)

        # --- Plot only top 3 tracks (thin red lines + circles) ---
        if not top3.empty:
            for _, t in top3.iterrows():
                ax.plot(
                    [t.f1_x, t.f2_x, t.f3_x],
                    [t.f1_y, t.f2_y, t.f3_y],
                    color="red", lw=0.8, alpha=0.4,
                )
                ax.scatter(
                    [t.f1_x, t.f2_x, t.f3_x],
                    [t.f1_y, t.f2_y, t.f3_y],
                    s=14, edgecolor="red", facecolor="none", lw=0.6, alpha=0.5,
                )

        ax.set_title(f"Frame {i+1}", fontsize=10, pad=3)
        ax.set_axis_off()
        ax.text(t.f3_x + 8, t.f3_y, f"#{int(t['rank'])}",
        color="red", fontsize=8, alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _draw_frame(ax, img, x, y, title="", radius=5, lw=0.5, alpha=0.20):
    img = np.asarray(img)
    ax.imshow(img, cmap="gray", origin="lower", interpolation="nearest")
    if np.isfinite(x) and np.isfinite(y):
        circ = Circle((float(x), float(y)), radius, fill=False,
                      edgecolor="red", linewidth=lw, alpha=alpha)
        ax.add_patch(circ)
        ax.text(0.02, 0.98, f"({x:.1f}, {y:.1f})",
                color="red", fontsize=8, ha="left", va="top",
                transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.30, edgecolor="none", pad=1.5))
    ax.set_title(title, fontsize=9, pad=2)
    ax.set_axis_off()

def show_triplet_row(F1, F2, F3,
                     f1x, f1y, f2x, f2y, f3x, f3y,
                     pixel_radius=5, fig_w=5.0, fig_h=1.6, dpi=150):
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=dpi)
    _draw_frame(axes[0], F1, f1x, f1y, "F1", radius=pixel_radius)
    _draw_frame(axes[1], F2, f2x, f2y, "F2", radius=pixel_radius)
    _draw_frame(axes[2], F3, f3x, f3y, "F3", radius=pixel_radius)
    st.pyplot(fig)
    plt.close(fig)


def render_triplet_preview(folder: Path, row: pd.Series):
    aligned_dir = folder / "aligned"
    fits_files = sorted(aligned_dir.glob("*_wcs_aligned.fits"))
    if len(fits_files) < 3:
        st.warning("No aligned FITS to preview.")
        return
    # Load all three
    with fits.open(fits_files[0]) as h1, fits.open(fits_files[1]) as h2, fits.open(fits_files[2]) as h3:
        F1 = h1[0].data; F2 = h2[0].data; F3 = h3[0].data
    # Show row
    show_triplet_row(
        F1, F2, F3,
        row.get("f1_x"), row.get("f1_y"),
        row.get("f2_x"), row.get("f2_y"),
        row.get("f3_x"), row.get("f3_y"),
        pixel_radius=5, fig_w=5.0, fig_h=1.6, dpi=150
    )

# -------------------------------
# UI
# -------------------------------
uploaded = st.file_uploader(
    "Upload a ZIP containing one asteroid folder (with either aligned *_wcs_aligned.fits or raw FITS + .wcs)",
    type=["zip"]
)
run_btn = st.button("Run pipeline")

if "df_scored" not in st.session_state:
    st.session_state.df_scored = None
if "work_folder" not in st.session_state:
    st.session_state.work_folder = None

if uploaded and run_btn:
    with st.spinner("Preparing input..."):
        folder = unzip_to_tmp(uploaded)
        st.session_state.work_folder = folder
        st.write(f"📂 Using folder: `{folder}`")

    try:
        # 1) Prefer aligned triplet; otherwise align from raw FITS + .wcs
        if have_aligned_triplet(folder):
            st.success("Found pre-aligned FITS (`*_wcs_aligned.fits`). Skipping alignment.")
        else:
            if have_raw_fits_plus_wcs(folder):
                st.info("No pre-aligned triplet. Found raw FITS + .wcs. Running alignment…")
                ok = align_folder(folder)
                if not ok or not have_aligned_triplet(folder):
                    st.error("Alignment did not produce 3 aligned frames. Aborting.")
                    st.stop()
                st.success("Alignment complete.")
            else:
                st.error("No aligned triplet and no valid raw FITS + .wcs sidecars. Please provide either.")
                st.stop()

        # 2) Run the full pipeline (candidate gen → SNR/Gmag → hybrid scoring)
        with st.spinner("Running detection → SNR/Gmag → hybrid scoring…"):
            out_root = folder
            df_scored = run_full_pipeline(folder, out_root)

        st.session_state.df_scored = df_scored
        st.success("Pipeline complete ✅")

    except Exception as e:
        st.error("Pipeline failed.")
        st.exception(e)
        st.stop()

# -------------------------------
# Results area
# -------------------------------
df = st.session_state.df_scored
if df is not None and len(df):
    st.subheader("Top-3 candidate tracks (by confidence score)")

    # Sort + take top 3 by hybrid score
    if "confidence_score" not in df.columns:
        st.error("Expected 'confidence_score' in results but did not find it.")
        st.stop()

    top3 = df.sort_values("confidence_score", ascending=False).head(3).copy()
    top3 = top3.reset_index(drop=True)
    top3["rank"] = top3.index + 1

    # ---- Resolve gmag & snr column aliases defensively ----
    def pick_col(cands):
        for c in cands:
            if c in top3.columns:
                return c
        return None

    g1 = pick_col(["gmag1","gmag_f1","gmag_f01","f1_gmag"])
    g2 = pick_col(["gmag2","gmag_f2","gmag_f02","f2_gmag"])
    g3 = pick_col(["gmag3","gmag_f3","gmag_f03","f3_gmag"])
    s1 = pick_col(["snr1","snr_f1","snr_f01","f1_snr","SNR1","SNR_f1"])
    s2 = pick_col(["snr2","snr_f2","snr_f02","f2_snr","SNR2","SNR_f2"])
    s3 = pick_col(["snr3","snr_f3","snr_f03","f3_snr","SNR3","SNR_f3"])

    # Build the display table with (x, y, SNR, gmag) per frame
    disp_rows = []
    for _, r in top3.iterrows():
        disp_rows.append({
            "Rank": int(r.get("rank", 0)),
            "Confidence Score": float(r.get("confidence_score", 0.0)),

            "F1_x": float(r.get("f1_x", float("nan"))),
            "F1_y": float(r.get("f1_y", float("nan"))),
            "F1_SNR": float(r.get(s1, float("nan"))) if s1 else float("nan"),
            "F1_gmag": float(r.get(g1, float("nan"))) if g1 else float("nan"),

            "F2_x": float(r.get("f2_x", float("nan"))),
            "F2_y": float(r.get("f2_y", float("nan"))),
            "F2_SNR": float(r.get(s2, float("nan"))) if s2 else float("nan"),
            "F2_gmag": float(r.get(g2, float("nan"))) if g2 else float("nan"),

            "F3_x": float(r.get("f3_x", float("nan"))),
            "F3_y": float(r.get("f3_y", float("nan"))),
            "F3_SNR": float(r.get(s3, float("nan"))) if s3 else float("nan"),
            "F3_gmag": float(r.get(g3, float("nan"))) if g3 else float("nan"),
        })

    disp_df = pd.DataFrame(disp_rows)

    with pd.option_context('display.precision', 3):
        st.dataframe(
            disp_df.assign(
                **({"Confidence Score": disp_df["Confidence Score"].round(3)} if "Confidence Score" in disp_df else {}),
                F1_x   = disp_df["F1_x"].round(2), F1_y = disp_df["F1_y"].round(2),
                F1_SNR = disp_df["F1_SNR"].round(2), F1_gmag = disp_df["F1_gmag"].round(1),
                F2_x   = disp_df["F2_x"].round(2), F2_y = disp_df["F2_y"].round(2),
                F2_SNR = disp_df["F2_SNR"].round(2), F2_gmag = disp_df["F2_gmag"].round(1),
                F3_x   = disp_df["F3_x"].round(2), F3_y = disp_df["F3_y"].round(2),
                F3_SNR = disp_df["F3_SNR"].round(2), F3_gmag = disp_df["F3_gmag"].round(1),
            ),
            hide_index=True,
            width="stretch"
        )

    # ---- The ONLY CSV download option (full hybrid-scored candidates) ----
    st.download_button(
        "⬇️ Download confidence-scored candidates (CSV)",
        data=df.to_csv(index=False),
        file_name="candidates_scored_confidence.csv",
        mime="text/csv",
        width="stretch"
    )

    with st.expander("Show all three aligned frames (Top-3 only)"):
        try:
            show_all_three_aligned_frames(st.session_state.work_folder, top3)

        except Exception as e:
            st.warning("Failed to render aligned frames.")
            st.exception(e)


            # --- MPC export (Top-3 only; no RA/Dec preview) ---
        # --- MPC export (combined Top-3) ---
    st.subheader("MPC export")

    colA, colB, colC = st.columns(3)
    with colA:
        mpc_designation = st.text_input("Designation", value="A10Wcxt")
    with colB:
        mpc_band = st.text_input("Photometric band", value="G")
    with colC:
        mpc_obs_code = st.text_input("Observatory code", value="M28")

    from Core_pipeline.mpc_export import build_mpc_for_track

    combined_mpc_text = ""
    for i in range(min(3, len(top3))):
        row = top3.iloc[i].copy()
        # fallback magnitude
        if "gmag_mean_obs" not in row.index:
            if "gmag" in top3.columns:
                row["gmag_mean_obs"] = row["gmag"]
            elif "gmag_mean" in top3.columns:
                row["gmag_mean_obs"] = row["gmag_mean"]
        try:
            txt = build_mpc_for_track(
                folder=st.session_state.work_folder,
                row=row,
                designation=mpc_designation,
                band=mpc_band,
                obs_code=mpc_obs_code,
            )
            combined_mpc_text += txt.strip() + "\n\n"
        except Exception as e:
            combined_mpc_text += f"# MPC export failed for Rank {i+1}: {e}\n\n"

    if combined_mpc_text.strip():
        st.download_button(
            "⬇️ Download MPC (Top-3 Combined)",
            data=combined_mpc_text.encode("utf-8"),
            file_name="MPC_submission.txt",
            mime="text/plain",
            use_container_width=True,
        )


else:
    st.info("Upload a ZIP and click **Run pipeline** to begin.")
