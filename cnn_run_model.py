#!/usr/bin/env python3
# cnn_run_model.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# ---------------- Dataset ----------------
class UnlabeledStampDataset(Dataset):
    """
    Loads 5-channel stamps from <root_dir>/test/*.npz saved by your 5-channel builder.
    Returns (x, relative_path_from_<root_dir>/.., group_name)
    """
    def __init__(self, root_dir: Path, group_map: dict):
        self.root = Path(root_dir)
        self.paths = sorted(p for p in self.root.rglob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz found in {self.root}")
        self.group_map = group_map

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        # rel path starting at "test/..."
        try:
            rel_from_root = p.relative_to(self.root).as_posix()
        except Exception:
            # fallback: best effort strip any leading folders up to 'test/'
            rel_from_root = str(p).replace("\\", "/")
            idx = rel_from_root.lower().rfind("/test/")
            if idx >= 0:
                rel_from_root = rel_from_root[idx + 1:]  # include "test/..."
        arr = np.load(p)["x"].astype(np.float32)  # (5,H,W) or (H,W,5)
        if arr.ndim == 3 and arr.shape[-1] == 5:
            arr = np.transpose(arr, (2, 0, 1))
        # robust per-channel standardization
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        m = arr.mean(axis=(1, 2), keepdims=True)
        s = arr.std(axis=(1, 2), keepdims=True)
        arr = np.clip((arr - m) / (s + 1e-6), -5, 5)
        x = torch.tensor(arr, dtype=torch.float32)

        group = self.group_map.get(rel_from_root, p.parent.name)
        return x, rel_from_root, group


def collate_keep_meta(batch):
    xs = torch.stack([x for x, _, _ in batch], dim=0)
    rel = [r for _, r, _ in batch]
    grp = [g for *_, g in batch]
    return xs, rel, grp


# ---------------- Model ----------------
class EfficientNet5(nn.Module):
    """
    EfficientNet-B0 backbone with 5-channel first conv.
    """
    def __init__(self):
        super().__init__()
        try:
            base = models.efficientnet_b0(weights=None)  # torchvision>=0.13
        except Exception:
            base = models.efficientnet_b0(pretrained=False)
        old = base.features[0][0]
        new = nn.Conv2d(
            5,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        base.features[0][0] = new
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        return x


# ---------------- Predict helper ----------------
@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device, tta: bool = True):
    """
    Returns: (probs: np.ndarray, rel_paths: List[str], groups: List[str])
    """
    model.eval()
    out_probs, out_paths, out_groups = [], [], []
    for xb, rels, groups in loader:
        xb = xb.to(device)
        if tta:
            p = torch.sigmoid(model(xb)).squeeze(1)
            p += torch.sigmoid(model(torch.flip(xb, dims=[-1]))).squeeze(1)
            p += torch.sigmoid(model(torch.flip(xb, dims=[-2]))).squeeze(1)
            p += torch.sigmoid(model(xb.transpose(2, 3))).squeeze(1)
            probs = (p / 4.0).detach().cpu().numpy()
        else:
            probs = torch.sigmoid(model(xb)).squeeze(1).detach().cpu().numpy()
        out_probs.append(probs)
        out_paths.extend(rels)
        out_groups.extend(groups)
    return np.concatenate(out_probs), out_paths, out_groups


# ---------------- Inference entrypoints ----------------
def run_cnn_on_stamps_root(
    stamps_root: Path,
    ckpt_path: Path,
    thr_path: Path | None = None,       # kept for signature compatibility; unused here
    candidate_csv: Path | None = None,  # kept for signature compatibility; unused here
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device | None = None,
    use_tta: bool = True,
) -> pd.DataFrame:
    """
    Legacy/batch inference over <stamps_root>/<folder>/test/*.npz.
    Returns: ['folder','f1_x','f1_y','f2_x','f2_y','f3_x','f3_y','cnn_prob']
    """
    return run_cnn_verify_for_dashboard(
        stamps_root=stamps_root,
        ckpt_path=ckpt_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        use_tta=use_tta,
        save_debug_dir=None,
    )


def run_cnn_verify_for_dashboard(
    stamps_root: Path,
    ckpt_path: Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device | None = None,
    use_tta: bool = True,
    save_debug_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Dashboard-friendly inference.
    Consumes prepared stamps + metadata and returns a tidy DataFrame:
      ['folder','f1_x','f1_y','f2_x','f2_y','f3_x','f3_y','cnn_prob']
    """
    root = Path(stamps_root)
    ckpt_path = Path(ckpt_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folders = [p for p in root.iterdir() if (p / "test").exists()]
    if not folders:
        return pd.DataFrame(columns=["folder","f1_x","f1_y","f2_x","f2_y","f3_x","f3_y","cnn_prob"])

    # Load model once
    model = EfficientNet5().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_frames: List[pd.DataFrame] = []

    with torch.no_grad():
        for ast_dir in folders:
            test_dir = ast_dir / "test"
            meta_csv = ast_dir / "metadata.csv"
            if not meta_csv.exists():
                # Can't merge back to coords without metadata; skip
                continue

            # group map from metadata (optional; group name = folder)
            group_map = {}
            try:
                meta_tmp = pd.read_csv(meta_csv)
                if "stamp_path" in meta_tmp.columns and "folder" in meta_tmp.columns:
                    for p_rel, g in zip(meta_tmp["stamp_path"], meta_tmp["folder"]):
                        group_map[str(p_rel).replace("\\", "/")] = str(g)
            except Exception:
                pass

            ds = UnlabeledStampDataset(test_dir, group_map)
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_keep_meta,
                pin_memory=(device.type == "cuda"),
            )

            probs, rel_paths, _groups = predict_probs(model, dl, device=device, tta=use_tta)
            

            pred_df = pd.DataFrame({"filename": rel_paths, "pred_prob": probs})

            # Build the SAME key as metadata: "<folder>/test/<file>.npz"
            pred_df["stamp_path"] = [
                f"{ast_dir.name}/test/{Path(p).name}".replace("\\", "/") for p in pred_df["filename"]
            ]

            meta = pd.read_csv(meta_csv)
            meta["stamp_path"] = meta["stamp_path"].astype(str).replace("\\", "/", regex=False)

            merged = pred_df.merge(
                meta[["stamp_path","folder","cx","cy","f1_x","f1_y","f2_x","f2_y","f3_x","f3_y"]],
                on="stamp_path",
                how="left",
            )


            keep = ["folder","f1_x","f1_y","f2_x","f2_y","f3_x","f3_y","pred_prob"]
            merged = merged[keep].dropna(subset=["folder"]).copy()

            if save_debug_dir is not None:
                out_dir = Path(save_debug_dir) / ast_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                merged.assign(folder_name=ast_dir.name).to_csv(out_dir / "preds.csv", index=False)

            out_frames.append(merged)

    if not out_frames:
        return pd.DataFrame(columns=["folder","f1_x","f1_y","f2_x","f2_y","f3_x","f3_y","cnn_prob"])

    out = pd.concat(out_frames, ignore_index=True)
    for k in ["f1_x","f1_y","f2_x","f2_y","f3_x","f3_y"]:
        if k in out.columns:
            out[k] = pd.to_numeric(out[k], errors="coerce")
    out = out.rename(columns={"pred_prob": "cnn_prob"}).drop_duplicates()
    return out


__all__ = [
    "UnlabeledStampDataset",
    "collate_keep_meta",
    "EfficientNet5",
    "predict_probs",
    "run_cnn_on_stamps_root",
    "run_cnn_verify_for_dashboard",
]
