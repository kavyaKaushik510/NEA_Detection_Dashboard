import pandas as pd
import joblib
import numpy as np


def predict_gb_on_df(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    X_cols = [
        'angle_change_deg','vel_ratio','disp12_px','disp23_px',
        'gmag1','gmag2','gmag3','gmag_mean','gmag_std'
    ]
    # derive SNR stats if missing
    for f in ['snr_mean','snr_std','snr_cv']:
        if f not in df.columns:
            df[f] = df[['snr1','snr2','snr3']].agg(
                {'snr_mean':'mean','snr_std':'std'}[f] if f!='snr_cv'
                else lambda r: r.std()/max(r.mean(),1e-9), axis=1
            )
    X = df[X_cols + ['snr_std']].fillna(-999)
    probs = model.predict_proba(X)[:, 1]
    out = df.copy()
    out['gb_confidence'] = probs
    return out


