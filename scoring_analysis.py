#!/usr/bin/env python3
"""
combine_and_rank_analysis.py
----------------------------
Combine geometric (GB) and CNN prediction CSVs, compute a hybrid score,
and evaluate whether the truth track ranks top-1 / top-3 under each score.
"""

import pandas as pd
import numpy as np

# ==================== CONFIG ====================
GEOM_CSV = "gb_2.csv"   # <-- first CSV (contains gb_confidence)
ML_CSV   = "cnn_2.csv"    # <-- second CSV (contains pred_prob)
OUT_CSV  = "combined_hybrid_results.csv"

# ==================== LOAD ====================
geom = pd.read_csv(GEOM_CSV)
ml   = pd.read_csv(ML_CSV)

geom.columns = geom.columns.str.strip().str.lower()
ml.columns   = ml.columns.str.strip().str.lower()

# ==================== MERGE ====================
merge_keys = ["folder","f1_x","f1_y","f2_x","f2_y","f3_x","f3_y"]
df = pd.merge(geom, ml, on=merge_keys, how="inner", suffixes=("_geom","_ml"))

# ==================== COLUMN DETECTION ====================
pred_col   = [c for c in df.columns if "pred_prob" in c][0]
gb_col     = [c for c in df.columns if "gb_confidence" in c][0]
truth_col  = [c for c in df.columns if "is_truth" in c][0]

print(f"Detected columns → pred_prob={pred_col}, gb_confidence={gb_col}, is_truth={truth_col}")

# ==================== HYBRID SCORE ====================
df["hybrid_score"] = 0.5 * df[pred_col] + 0.7 * df[gb_col]

# ==================== SELECT OUTPUT COLUMNS ====================
cols_out = [
    "folder", "f1", "f2", "f3",
    "gmag1", "gmag2", "gmag3",
    truth_col, pred_col, gb_col, "hybrid_score"
]
df_out = df[cols_out].copy()

# ==================== FILTER FOLDERS WITH ≥1 TRUTH ====================
folders_with_truth = df_out.groupby("folder")[truth_col].any()
valid_folders = folders_with_truth[folders_with_truth].index
df_valid = df_out[df_out["folder"].isin(valid_folders)]

# ==================== RANKING ANALYSIS ====================
def rank_check(group, score_col):
    """Return True if a truth track is in top-1 or top-3 for given score."""
    g_sorted = group.sort_values(score_col, ascending=False).reset_index(drop=True)
    truth_rows = g_sorted[g_sorted[truth_col] == True]
    if truth_rows.empty:
        return pd.Series({"top1": False, "top3": False})
    truth_rank = truth_rows.index.min()
    return pd.Series({
        "top1": truth_rank == 0,
        "top3": truth_rank < 3
    })

results = []
for score_col in [pred_col, gb_col, "hybrid_score"]:
    r = df_valid.groupby("folder").apply(rank_check, score_col=score_col)
    summary = {
        "score_type": score_col,
        "folders": len(r),
        "top1_count": int(r["top1"].sum()),
        "top3_count": int(r["top3"].sum()),
        "top1_pct": 100 * r["top1"].mean(),
        "top3_pct": 100 * r["top3"].mean(),
    }
    results.append(summary)

summary_df = pd.DataFrame(results)

# ==================== SAVE & DISPLAY ====================
df_out.to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved merged output: {OUT_CSV}\n")

print("📊 Ranking Summary (folders with ≥1 truth):")
print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.1f}'))
