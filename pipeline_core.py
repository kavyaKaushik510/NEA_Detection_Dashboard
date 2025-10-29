# pipeline_core.py
from pathlib import Path
import pandas as pd
import time

from alignment import align_folder
from generate_candidates_copy_v3 import run_pipeline_single
from five_channel_v2 import build_test_stamps_for_candidates
from cnn_run_model import run_cnn_verify_for_dashboard
from test_2_gb import predict_gb_on_df
from scoring import combine_cnn_gb
from append_s_g import augment_snr_gmag_df
from convert_to_wcs import export_top3_radec_csv
from mpc_export import build_mpc_for_track

MERGE_KEYS = ["folder","f1_x","f1_y","f2_x","f2_y","f3_x","f3_y"]

def _canon_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "folder" in df.columns:
        df["folder"] = df["folder"].astype(str)
    for k in ["f1_x","f1_y","f2_x","f2_y","f3_x","f3_y"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("float64")
    return df


def run_full_pipeline(
    folder: Path,
    output_dir: Path,
    *,
    gb_model_path: Path = Path("ml_results/asteroid_gb_model.pkl"),
    ckpt_path: Path = Path("models/best_model.pt"),
    stamps_subdir: str = "stamps_test",
    stamp_size: int = 64,
) -> pd.DataFrame:
    """
    End-to-end for a single folder with timing logs.
    """
    folder = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_csv = output_dir / "all_candidates.csv"
    log_path = output_dir / "pipeline_timing.txt"

    # Helper to record timing
    def log_stage(stage_name, start_time, end_time):
        elapsed = end_time - start_time
        msg = f"{stage_name}: {elapsed:.2f} s\n"
        print(msg.strip())
        with open(log_path, "a") as f:
            f.write(msg)

    # Clear old timing file
    if log_path.exists():
        log_path.unlink()

    total_start = time.time()

    # 1️⃣ Alignment
    t0 = time.time()
    print("1️⃣ Alignment…")
    align_folder(folder)
    log_stage("Alignment", t0, time.time())

    # 2️⃣ Candidate generation
    t0 = time.time()
    print("2️⃣ Candidate generation…")
    run_pipeline_single(folder, master_csv)
    log_stage("Candidate generation", t0, time.time())

    # 3️⃣ Photometry
    t0 = time.time()
    print("3️⃣ SNR + Gmag computation…")
    df = pd.read_csv(master_csv)
    df = augment_snr_gmag_df(df, folder=folder)
    log_stage("Photometry (SNR + Gmag)", t0, time.time())

    print("=== BEFORE SCORING: columns ===")
    print(sorted(df.columns))

    # 4️⃣ GB + CNN scoring
    print("4️⃣ Hybrid scoring (GB + CNN)…")

    # 4a) Gradient Boosting
    t0 = time.time()
    df_gb = predict_gb_on_df(df, str(gb_model_path))
    df_gb.to_csv("gb_scored.csv", index=False)
    log_stage("Gradient Boosting scoring", t0, time.time())

    # 4b) CNN scoring
    t0 = time.time()
    stamps_root = output_dir / stamps_subdir
    build_test_stamps_for_candidates(folder, master_csv, stamps_root, stamp=stamp_size)
    df_cnn = run_cnn_verify_for_dashboard(
        stamps_root=stamps_root,
        ckpt_path=ckpt_path,
        batch_size=64,
        num_workers=0,
        use_tta=True,
    )
    df_cnn.to_csv("cnn_scored.csv", index=False)
    log_stage("CNN scoring", t0, time.time())

    # 4c) Combine
    t0 = time.time()
    df_hcore = combine_cnn_gb(df_gb, df_cnn, w_cnn=0.6, w_gb=0.4)
    df = _canon_merge_keys(df)
    df_hcore = _canon_merge_keys(df_hcore)
    merge_on = [k for k in MERGE_KEYS if k in df.columns]
    df_scored = df.merge(
        df_hcore[merge_on + ["cnn_prob","gb_confidence","hybrid_score"]].drop_duplicates(merge_on),
        on=merge_on, how="left", validate="1:1",
    )
    df_scored["confidence_score"] = df_scored["hybrid_score"].fillna(0.0)
    log_stage("Hybrid score merge", t0, time.time())

    # 5️⃣ Export final CSV
    t0 = time.time()
    df_sorted = df_scored.sort_values("confidence_score", ascending=False).copy()
    df_output = output_dir / "candidates_scored_hybrid.csv"
    df_sorted.to_csv(df_output, index=False)
    log_stage("Export final CSV", t0, time.time())

    total_end = time.time()
    log_stage("TOTAL PIPELINE TIME", total_start, total_end)

    print(f"✅ Hybrid-scored candidates saved to {df_output}")
    print(f"🕒 Timing summary written to {log_path}")

    return df_sorted
