# task_weather_time_only.py
# Predict ONLY: Weather + Time of Day from images
# Models:
#  - Task 3 baseline: kNN on ResNet18 embeddings (k=1,3)
#  - Task 4 Model A: Linear SVM (MultiOutput) on embeddings (tune C with 4 values)
#  - Task 4 Model B: Fine-tuned ResNet18 Multi-Head CNN (tune lr with 4 values)
#  - Task 5: Error tables + confusion pattern summaries for each model

import os
import hashlib
import requests
from io import BytesIO
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Optional AVIF support (install: pip install pillow-avif-plugin)
try:
    from pillow_avif import AvifImagePlugin  # noqa: F401
except Exception:
    pass


# =========================
# Config (EDIT THIS PATH)
# =========================
CSV_PATH = r"C:/Users/Asus/OneDrive/Documents/cleaned_travel_data_98989898.csv"
IMAGE_CACHE_DIR = "image_cache"
RANDOM_SEED = 42

# ONLY TWO TARGETS
TARGET_COLS = ["Weather", "Time of Day"]

# Task 3 baseline kNN
K_VALUES = [1, 3]

# Task 4 tuning grids (>=4 values each)
SVM_C_VALUES = [0.01, 0.1, 1, 10]
CNN_LR_VALUES = [1e-4, 3e-4, 1e-3, 3e-3]

# Training knobs
CNN_EPOCHS = 3
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows safe

MIN_IMAGE_SIZE = 80  # skip tiny images


# =========================
# Reproducibility
# =========================
def seed_everything(seed=RANDOM_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Image caching
# =========================
def url_to_filename(url: str) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return f"{h}.jpg"


def download_image(url: str, cache_dir: str = IMAGE_CACHE_DIR, timeout: int = 20) -> str | None:
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, url_to_filename(url))

    # reuse cached
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")

        w, h = img.size
        if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
            return None

        img.save(local_path, format="JPEG", quality=90)
        return local_path
    except Exception:
        return None


# =========================
# ResNet18 embedder (for kNN & SVM)
# =========================
def build_resnet18_embedder(device: str):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)

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
def image_to_embedding(img_path: str, model, preprocess, device: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    emb = model(x).squeeze(0).cpu().numpy().astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


def cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    sim = A @ B.T
    return 1.0 - sim


def knn_predict_multioutput(X_train: np.ndarray, y_train: pd.DataFrame, X_test: np.ndarray, k: int) -> pd.DataFrame:
    dists = cosine_distance_matrix(X_test, X_train)
    nn_idx = np.argsort(dists, axis=1)[:, :k]

    preds = {}
    for col in y_train.columns:
        col_train = y_train[col].values
        pred_col = []
        for neighbors in nn_idx:
            neigh_labels = col_train[neighbors]
            vals, counts = np.unique(neigh_labels, return_counts=True)
            pred_col.append(vals[np.argmax(counts)])
        preds[col] = pred_col

    return pd.DataFrame(preds)


# =========================
# Metrics + Task 5 helpers
# =========================
def exact_match(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    return (y_true.values == y_pred.values).all(axis=1).mean()


def print_multioutput_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    ex = exact_match(y_true, y_pred)
    print(f"Exact Match Accuracy (both labels correct): {ex:.4f}")

    for col in y_true.columns:
        acc = accuracy_score(y_true[col], y_pred[col])
        f1 = f1_score(y_true[col], y_pred[col], average="macro", zero_division=0)
        print(f"{col:12s} | Acc: {acc:.4f} | Macro-F1: {f1:.4f}")

    return ex


def save_error_table(df_meta: pd.DataFrame, y_true: pd.DataFrame, y_pred: pd.DataFrame, out_csv: str):
    out = df_meta.copy().reset_index(drop=True)
    for col in y_true.columns:
        out[f"true_{col}"] = y_true[col].values
        out[f"pred_{col}"] = y_pred[col].values
        out[f"ok_{col}"] = (y_true[col].values == y_pred[col].values)
    out["exact_ok"] = out[[f"ok_{c}" for c in y_true.columns]].all(axis=1)
    out.to_csv(out_csv, index=False)
    print(f"\nSaved error table: {out_csv}")
    return out


def task5_analyze_errors(error_csv: str):
    df = pd.read_csv(error_csv)
    print("\n" + "=" * 90)
    print(f"TASK 5 ERROR ANALYSIS: {error_csv}")
    print("=" * 90)

    print(f"Exact match (from file): {df['exact_ok'].mean():.4f}")

    # label failure rates
    print("\nLabel failure rates (higher = harder):")
    for col in TARGET_COLS:
        print(f"  {col:12s}: {(1.0 - df[f'ok_{col}'].mean()):.3f}")

    # top confusion pairs
    print("\nTop confusion pairs per label (true -> pred):")
    for col in TARGET_COLS:
        tcol, pcol = f"true_{col}", f"pred_{col}"
        wrong = df[df[tcol] != df[pcol]]

        conf = Counter(zip(wrong[tcol].astype(str), wrong[pcol].astype(str)))
        print(f"\n{col}:")
        for (t, p), c in conf.most_common(10):
            print(f"  {t} -> {p}: {c}")


# =========================
# CNN dataset + model (2 heads)
# =========================
def get_cnn_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=models.ResNet18_Weights.DEFAULT.transforms().mean,
            std=models.ResNet18_Weights.DEFAULT.transforms().std
        ),
    ])


class TravelDataset2(Dataset):
    def __init__(self, df: pd.DataFrame, encoders: dict, transform):
        self.df = df.reset_index(drop=True)
        self.encoders = encoders
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["local_path"]).convert("RGB")
        x = self.transform(img)

        y_weather = self.encoders["Weather"].transform([row["Weather"]])[0]
        y_time = self.encoders["Time of Day"].transform([row["Time of Day"]])[0]
        return x, int(y_weather), int(y_time)


class MultiHeadResNet18_2(nn.Module):
    def __init__(self, n_weather, n_time, freeze_backbone=False):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # [B,512,1,1]
        self.feat_dim = 512

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head_weather = nn.Linear(self.feat_dim, n_weather)
        self.head_time = nn.Linear(self.feat_dim, n_time)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.head_weather(f), self.head_time(f)


def train_one_cnn(lr, train_loader, val_loader, encoders, device, epochs=CNN_EPOCHS):
    n_weather = len(encoders["Weather"].classes_)
    n_time = len(encoders["Time of Day"].classes_)

    model = MultiHeadResNet18_2(n_weather, n_time, freeze_backbone=False).to(device)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_exact = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for x, y_w, y_t in train_loader:
            x = x.to(device)
            y_w = y_w.to(device)
            y_t = y_t.to(device)

            opt.zero_grad()
            o_w, o_t = model(x)
            loss = ce(o_w, y_w) + ce(o_t, y_t)
            loss.backward()
            opt.step()

        # validate
        model.eval()
        all_true_w, all_true_t = [], []
        all_pred_w, all_pred_t = [], []

        with torch.no_grad():
            for x, y_w, y_t in val_loader:
                x = x.to(device)
                y_w = y_w.to(device)
                y_t = y_t.to(device)

                o_w, o_t = model(x)
                p_w = torch.argmax(o_w, dim=1)
                p_t = torch.argmax(o_t, dim=1)

                all_true_w.extend(y_w.cpu().numpy().tolist())
                all_true_t.extend(y_t.cpu().numpy().tolist())
                all_pred_w.extend(p_w.cpu().numpy().tolist())
                all_pred_t.extend(p_t.cpu().numpy().tolist())

        ytrue = np.vstack([all_true_w, all_true_t]).T
        ypred = np.vstack([all_pred_w, all_pred_t]).T
        val_exact = (ytrue == ypred).all(axis=1).mean()
        print(f"[CNN lr={lr} ep={ep}/{epochs}] val exact: {val_exact:.4f}")

        if val_exact > best_val_exact:
            best_val_exact = val_exact
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_exact


def cnn_predict(model, loader, encoders, device):
    model.eval()
    all_true_w, all_true_t = [], []
    all_pred_w, all_pred_t = [], []

    with torch.no_grad():
        for x, y_w, y_t in loader:
            x = x.to(device)
            y_w = y_w.to(device)
            y_t = y_t.to(device)

            o_w, o_t = model(x)
            p_w = torch.argmax(o_w, dim=1)
            p_t = torch.argmax(o_t, dim=1)

            all_true_w.extend(y_w.cpu().numpy().tolist())
            all_true_t.extend(y_t.cpu().numpy().tolist())
            all_pred_w.extend(p_w.cpu().numpy().tolist())
            all_pred_t.extend(p_t.cpu().numpy().tolist())

    y_true_df = pd.DataFrame({
        "Weather": encoders["Weather"].inverse_transform(np.array(all_true_w, dtype=int)),
        "Time of Day": encoders["Time of Day"].inverse_transform(np.array(all_true_t, dtype=int)),
    })
    y_pred_df = pd.DataFrame({
        "Weather": encoders["Weather"].inverse_transform(np.array(all_pred_w, dtype=int)),
        "Time of Day": encoders["Time of Day"].inverse_transform(np.array(all_pred_t, dtype=int)),
    })
    return y_true_df, y_pred_df


# =========================
# Main
# =========================
def main():
    seed_everything()
    df = pd.read_csv(CSV_PATH)

    # Download/cache images (reuses cache on reruns)
    print("Downloading images (and caching)...")
    local_paths, keep_mask = [], []

    for url in tqdm(df["Image URL"].astype(str).tolist()):
        url = url.strip()
        p = download_image(url)
        if p is None:
            keep_mask.append(False)
            local_paths.append(None)
        else:
            keep_mask.append(True)
            local_paths.append(p)

    df = df.loc[keep_mask].copy()
    df["local_path"] = [p for p in local_paths if p is not None]
    df = df.reset_index(drop=True)
    print(f"Rows kept: {len(df)}")

    # Outer split: train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
    # Inner split: train/val (for tuning)
    train_df2, val_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

    y_train = train_df2[TARGET_COLS].copy()
    y_val   = val_df[TARGET_COLS].copy()
    y_test  = test_df[TARGET_COLS].copy()

    meta_test = test_df[["Image URL", "Description", "Country", "Activity"]].copy()

    # =========================================================
    # Task 3: Baseline kNN
    # =========================================================
    print("\n" + "=" * 90)
    print("TASK 3 BASELINE: kNN on ResNet18 embeddings (Weather + Time of Day)")
    print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder, preprocess = build_resnet18_embedder(device=device)

    print("Extracting embeddings...")
    X_train = np.vstack([image_to_embedding(p, embedder, preprocess, device) for p in tqdm(train_df2["local_path"])])
    X_val   = np.vstack([image_to_embedding(p, embedder, preprocess, device) for p in tqdm(val_df["local_path"])])
    X_test  = np.vstack([image_to_embedding(p, embedder, preprocess, device) for p in tqdm(test_df["local_path"])])

    best_k = None
    best_val_exact = -1.0
    best_knn_pred = None

    for k in K_VALUES:
        val_pred = knn_predict_multioutput(X_train, y_train, X_val, k=k)
        val_exact = print_multioutput_metrics(y_val, val_pred, title=f"kNN (k={k}) - VALIDATION")
        if val_exact > best_val_exact:
            best_val_exact = val_exact
            best_k = k

    test_pred_knn = knn_predict_multioutput(X_train, y_train, X_test, k=best_k)
    print_multioutput_metrics(y_test, test_pred_knn, title=f"kNN BEST (k={best_k}) - TEST")
    knn_err_csv = f"errors_knn_k{best_k}.csv"
    save_error_table(meta_test, y_test, test_pred_knn, knn_err_csv)
    task5_analyze_errors(knn_err_csv)

    # =========================================================
    # Task 4 Model A: SVM
    # =========================================================
    print("\n" + "=" * 90)
    print("TASK 4 MODEL A: Linear SVM (MultiOutput) on embeddings")
    print("=" * 90)

    best_C = None
    best_val_exact = -1.0
    best_svm = None

    for C in SVM_C_VALUES:
        svm = MultiOutputClassifier(LinearSVC(C=C, max_iter=5000, dual="auto"))
        svm.fit(X_train, y_train)
        val_pred = pd.DataFrame(svm.predict(X_val), columns=TARGET_COLS)
        val_exact = print_multioutput_metrics(y_val, val_pred, title=f"SVM (C={C}) - VALIDATION")
        if val_exact > best_val_exact:
            best_val_exact = val_exact
            best_C = C
            best_svm = svm

    test_pred_svm = pd.DataFrame(best_svm.predict(X_test), columns=TARGET_COLS)
    print_multioutput_metrics(y_test, test_pred_svm, title=f"SVM BEST (C={best_C}) - TEST")
    svm_err_csv = f"errors_svm_C{best_C}.csv"
    save_error_table(meta_test, y_test, test_pred_svm, svm_err_csv)
    task5_analyze_errors(svm_err_csv)

    # =========================================================
    # Task 4 Model B: CNN fine-tuning (2 heads)
    # =========================================================
    print("\n" + "=" * 90)
    print("TASK 4 MODEL B: Fine-tuned ResNet18 (2-head CNN)")
    print("=" * 90)

    # Label encoders
    encoders = {}
    for col in TARGET_COLS:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))  # fit on outer-train split
        encoders[col] = le

    transform = get_cnn_transform()

    train_ds = TravelDataset2(train_df2, encoders, transform)
    val_ds   = TravelDataset2(val_df, encoders, transform)
    test_ds  = TravelDataset2(test_df, encoders, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device_t = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CNN training device: {device_t}")

    best_lr = None
    best_val_exact = -1.0
    best_cnn = None

    for lr in CNN_LR_VALUES:
        model, val_exact = train_one_cnn(lr, train_loader, val_loader, encoders, device_t, epochs=CNN_EPOCHS)
        if val_exact > best_val_exact:
            best_val_exact = val_exact
            best_lr = lr
            best_cnn = model

    y_true_cnn, y_pred_cnn = cnn_predict(best_cnn, test_loader, encoders, device_t)
    print_multioutput_metrics(y_true_cnn, y_pred_cnn, title=f"CNN BEST (lr={best_lr}) - TEST")
    cnn_err_csv = f"errors_cnn_lr{best_lr}.csv"
    save_error_table(meta_test, y_true_cnn, y_pred_cnn, cnn_err_csv)
    task5_analyze_errors(cnn_err_csv)

    print("\nDONE ✅")
    print(f"- kNN errors: {knn_err_csv}")
    print(f"- SVM errors: {svm_err_csv}")
    print(f"- CNN errors: {cnn_err_csv}")


if __name__ == "__main__":
    main()
