# tempCodeRunnerFile.py (FULL WORKING CODE)
# Task 3 baseline: ResNet18 embeddings + kNN (k=1,3) for multi-output labels
# Includes: image caching + AVIF support + robust download + resizing/cropping/normalization

import os
import hashlib
import requests
from io import BytesIO

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torchvision import models, transforms

# Enable AVIF decoding in Pillow (after: pip install pillow-avif-plugin)
from pillow_avif import AvifImagePlugin  # noqa: F401


# =========================
# Config
# =========================
CSV_PATH = r"cleaned_travel_data_98989898.csv"
IMAGE_CACHE_DIR = "image_cache"
RANDOM_SEED = 42

TARGET_COLS = ["Weather", "Time of Day"]
# Expanded search for better performance
K_VALUES = [1, 3, 5, 7, 9, 11, 15]

# Optional: skip very tiny images (keep more rows by lowering this)
MIN_IMAGE_SIZE = 80  # pixels


# =========================
# Utilities
# =========================
def url_to_filename(url: str) -> str:
    """Stable filename for a URL using md5 hash."""
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return f"{h}.jpg"


def download_image(url: str, cache_dir: str = IMAGE_CACHE_DIR, timeout: int = 20) -> str | None:
    """
    Download image from URL and cache as JPEG.
    Returns local filepath if ok, else None.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, url_to_filename(url))

    # Reuse cache if present
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        # Decode image (AVIF works if pillow-avif-plugin is installed)
        img = Image.open(BytesIO(r.content)).convert("RGB")

        # Optional quality/size guard
        w, h = img.size
        if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
            return None

        # Save as JPEG to standardize format for downstream code
        img.save(local_path, format="JPEG", quality=90)
        return local_path

    except Exception:
        return None


def build_resnet18_embedder(device: str = "cpu"):
    """
    Pretrained ResNet18 -> embedding vector (512-d) by removing the classifier head.
    Includes required resizing/cropping/normalization.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)

    # ResNet/ImageNet standard preprocessing:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=models.ResNet18_Weights.DEFAULT.transforms().mean,
            std=models.ResNet18_Weights.DEFAULT.transforms().std
        ),
    ])
    return model, preprocess


@torch.no_grad()
def image_to_embedding(img_path: str, model, preprocess, device: str = "cpu") -> np.ndarray:
    """Convert a cached image file into a normalized embedding vector."""
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    emb = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)

    # L2 normalize for cosine distance usage
    norm = np.linalg.norm(emb) + 1e-12
    return emb / norm


def cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine distance = 1 - cosine similarity (works because embeddings are L2-normalized)."""
    sim = A @ B.T
    return 1.0 - sim


def knn_predict_multioutput(X_train: np.ndarray, y_train: pd.DataFrame, X_test: np.ndarray, k: int) -> pd.DataFrame:
    """
    Multi-output kNN using cosine distance + weighted voting.
    Weights = 1 - cosine_distance (which is cosine similarity).
    """
    dists = cosine_distance_matrix(X_test, X_train)
    # Get indices of k nearest neighbors
    nn_idx = np.argsort(dists, axis=1)[:, :k]

    # Get their distances to compute weights
    row_idx = np.arange(dists.shape[0])[:, None]
    nn_dists = dists[row_idx, nn_idx]
    
    # Weights = Cosine Similarity = 1 - Cosine Distance
    weights = 1.0 - nn_dists
    weights = np.maximum(weights, 0.0)

    preds = {}
    for col in y_train.columns:
        col_train = y_train[col].values
        pred_col = []
        
        for i, neighbors in enumerate(nn_idx):
            neigh_labels = col_train[neighbors]
            w = weights[i]
            
            # Weighted vote
            unique_labels = np.unique(neigh_labels)
            label_scores = {}
            for label in unique_labels:
                mask = (neigh_labels == label)
                label_scores[label] = np.sum(w[mask])
            
            best_label = max(label_scores, key=label_scores.get)
            pred_col.append(best_label)
            
        preds[col] = pred_col

    return pd.DataFrame(preds)


def evaluate_multioutput(y_true: pd.DataFrame, y_pred: pd.DataFrame, title: str = "") -> float:
    """
    Prints:
    - exact match accuracy (all 4 labels correct)
    - per-label accuracy and macro-F1
    Returns exact match accuracy.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    exact = (y_true.values == y_pred.values).all(axis=1).mean()
    print(f"Exact Match Accuracy (all {y_true.shape[1]} labels correct): {exact:.4f}")

    for col in y_true.columns:
        acc = accuracy_score(y_true[col], y_pred[col])
        f1 = f1_score(y_true[col], y_pred[col], average="macro", zero_division=0)
        print(f"{col:12s} | Acc: {acc:.4f} | Macro-F1: {f1:.4f}")

    return float(exact)


def build_error_table(df_test: pd.DataFrame, y_true: pd.DataFrame, y_pred: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    """
    Save a detailed error table for Task 5-style analysis.
    """
    out = df_test.copy().reset_index(drop=True)

    for col in y_true.columns:
        out[f"true_{col}"] = y_true[col].values
        out[f"pred_{col}"] = y_pred[col].values
        out[f"ok_{col}"] = (y_true[col].values == y_pred[col].values)

    out["exact_ok"] = out[[f"ok_{c}" for c in y_true.columns]].all(axis=1)
    out.to_csv(out_csv, index=False)
    print(f"\nSaved error table to: {out_csv}")
    return out


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(CSV_PATH)

    # sanity checks
    if "Image URL" not in df.columns:
        raise ValueError("CSV must contain 'Image URL' column.")

    missing_targets = [c for c in TARGET_COLS if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in CSV: {missing_targets}")

    # Download/cache images
    print("Downloading images (and caching)...")
    local_paths = []
    keep_mask = []

    for url in tqdm(df["Image URL"].astype(str).tolist()):
        url = url.strip()
        p = download_image(url)
        if p is None:
            keep_mask.append(False)
            local_paths.append(None)
        else:
            keep_mask.append(True)
            local_paths.append(p)

    total = len(df)
    kept = sum(keep_mask)
    dropped = total - kept

    df = df.loc[keep_mask].copy()
    df["local_path"] = [p for p in local_paths if p is not None]
    df = df.reset_index(drop=True)

    print(f"Rows total: {total}")
    print(f"Rows kept after image download+decode check: {kept}")
    print(f"Rows dropped (broken/unreadable/tiny): {dropped}")

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

    y_train = train_df[TARGET_COLS].copy()
    y_test = test_df[TARGET_COLS].copy()

    # Embedding extraction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = build_resnet18_embedder(device=device)

    print(f"\nExtracting embeddings on device: {device}")
    X_train = np.vstack([
        image_to_embedding(p, model, preprocess, device=device)
        for p in tqdm(train_df["local_path"].tolist())
    ])
    X_test = np.vstack([
        image_to_embedding(p, model, preprocess, device=device)
        for p in tqdm(test_df["local_path"].tolist())
    ])

    # kNN baseline
    best_k = None
    best_exact = -1.0
    best_pred = None

    for k in K_VALUES:
        y_pred = knn_predict_multioutput(X_train, y_train, X_test, k=k)
        exact = evaluate_multioutput(y_test, y_pred, title=f"Task 3 Baseline: kNN (cosine) with k={k}")

        if exact > best_exact:
            best_exact = exact
            best_k = k
            best_pred = y_pred

    # Error table (Task 5-style output)
    _ = build_error_table(
        df_test=test_df[["Image URL", "Description", "Country", "Activity"]].copy(),
        y_true=y_test,
        y_pred=best_pred,
        out_csv=f"baseline_knn_k{best_k}_errors.csv"
    )

    print(f"\nBest baseline (by exact match) was k={best_k} with Exact Match={best_exact:.4f}")


if __name__ == "__main__":
    main()
