import pandas as pd

MERGE_KEYS = ["folder","f1_x","f1_y","f2_x","f2_y","f3_x","f3_y"]

def _force_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "folder" in df.columns:
        df["folder"] = df["folder"].astype(str)
    for k in ["f1_x","f1_y","f2_x","f2_y","f3_x","f3_y"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("float64")
    return df

def combine_cnn_gb(df_gb: pd.DataFrame, df_cnn: pd.DataFrame,
                   w_cnn: float = 0.6, w_gb: float = 0.4) -> pd.DataFrame:
    # Lowercase columns for safety
    A = df_gb.copy(); A.columns = A.columns.str.lower()
    B = df_cnn.copy(); B.columns = B.columns.str.lower()

    # Standardize prob column names
    if "gb_confidence" not in A.columns:
        if "gb_prob" in A.columns: A = A.rename(columns={"gb_prob": "gb_confidence"})
        elif "gb_score" in A.columns: A = A.rename(columns={"gb_score": "gb_confidence"})
        else: A["gb_confidence"] = 0.0

    if "cnn_prob" not in B.columns:
        if "pred_prob" in B.columns: B = B.rename(columns={"pred_prob": "cnn_prob"})
        elif "prob" in B.columns:    B = B.rename(columns={"prob": "cnn_prob"})
        else: B["cnn_prob"] = 0.0

    # Enforce exact-match dtypes on keys
    A = _force_key_dtypes(A)
    B = _force_key_dtypes(B)

    # Keep only what we need from CNN side
    B_keep = B[MERGE_KEYS + ["cnn_prob"]].drop_duplicates(MERGE_KEYS)

    # Exact inner merge on the 7 keys
    df = pd.merge(
        A, B_keep,
        on=MERGE_KEYS,
        how="inner",           # exact match only
        validate="m:1"         # many GB rows, at most one CNN row per key
    )

    # Ensure probs are numeric in [0,1]
    df["gb_confidence"] = pd.to_numeric(df["gb_confidence"], errors="coerce").fillna(0.0).clip(0,1)
    df["cnn_prob"]      = pd.to_numeric(df["cnn_prob"],      errors="coerce").fillna(0.0).clip(0,1)

    # Hybrid: weighted sum (set to product if you truly want multiplicative)
    df["hybrid_score"] = w_cnn*df["cnn_prob"] + w_gb*df["gb_confidence"]
    # If you actually want weighted "multiplication", swap the line above for:
    # df["hybrid_score"] = (df["cnn_prob"]**w_cnn) * (df["gb_confidence"]**w_gb)

    return df
