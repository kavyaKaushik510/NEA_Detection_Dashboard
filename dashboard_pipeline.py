# dashboard.py
import io
from pathlib import Path

import pandas as pd
import streamlit as st

# Import your pipeline (assumes pipeline_core.py is in the same folder)
from pipeline_core import run_full_pipeline


st.set_page_config(page_title="Asteroid Pipeline Dashboard", layout="wide")
st.title("🚀 Asteroid Candidate Scoring — Hybrid (GB + CNN)")

with st.sidebar:
    st.header("Run Settings")

    # Required paths
    folder = st.text_input(
        "Target folder (with FITS & aligned/)",
        value=str(Path.cwd() / "sample_field"),  # change default if you like
        help="Path to the single field folder you want to process.",
    )
    out_root = st.text_input(
        "Output directory",
        value=str(Path.cwd() / "out"),
        help="Where the pipeline will write outputs (CSV + stamps).",
    )

    # Optional: model paths (override defaults in pipeline_core if needed)
    gb_model_path = st.text_input(
        "GB model (.pkl)",
        value=str(Path("ml_results/asteroid_gb_model.pkl")),
        help="Gradient Boosting model path.",
    )
    ckpt_path = st.text_input(
        "CNN checkpoint (.pt)",
        value=str(Path("models/best_model.pt")),
        help="CNN checkpoint path.",
    )

    run_btn = st.button("Run pipeline", type="primary",  width="stretch")

# Main area
placeholder = st.empty()

def load_existing_csv(out_dir: Path) -> pd.DataFrame | None:
    csv_path = out_dir / "candidates_scored_hybrid.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

# If user presses the run button, execute pipeline and show CSV
if run_btn:
    try:
        target = Path(folder).resolve()
        out_dir = Path(out_root).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Running full pipeline… this will align, generate candidates, score (GB+CNN), and produce the final CSV."):
            df_scored = run_full_pipeline(
                folder=target,
                output_dir=out_dir,
                gb_model_path=Path(gb_model_path),
                ckpt_path=Path(ckpt_path),
            )

        st.success("Done! Final hybrid-scored CSV has been created.")
        st.subheader("Hybrid-scored candidates (top rows)")
        st.dataframe(df_scored.head(500),  width="stretch")

        # Download the exact CSV that was saved by the pipeline
        csv_bytes = df_scored.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name="candidates_scored_hybrid.csv",
            mime="text/csv",
            width="stretch",
        )

    except Exception as e:
        st.error("Pipeline failed. See error below.")
        st.exception(e)
else:
    # If not running now, but a prior run exists, show it for convenience
    existing = load_existing_csv(Path(out_root))
    if existing is not None and not existing.empty:
        st.info("Showing existing results found in the output directory.")
        st.subheader("Hybrid-scored candidates (existing)")
        st.dataframe(existing.head(500),  width="stretch")

        csv_bytes = existing.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download existing results CSV",
            data=csv_bytes,
            file_name="candidates_scored_hybrid.csv",
            mime="text/csv",
            width="stretch"
        )
    else:
        st.info("Set the paths in the sidebar and click **Run pipeline** to generate the final CSV.")
