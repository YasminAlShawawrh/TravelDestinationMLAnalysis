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
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

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
# Config
# =========================
CSV_PATH = r"C:/Users/Asus/OneDrive/Documents/cleaned_travel_data_98989898.csv"
IMAGE_CACHE_DIR = "image_cache"
RANDOM_SEED = 42

TARGET_COLS = ["Weather", "Time of Day"]

# Task 4 tuning
SVM_C_VALUES = [0.01, 0.1, 1, 10]                  # 4 values ✅
CNN_LR_VALUES = [1e-4, 3e-4, 1e-3, 3e-3]           # 4 values ✅

# Training speed knobs (you can increase epochs later)
# Training speed knobs (you can increase epochs later)
CNN_EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows: keep 0 if you get DataLoader issues

# Minimum image size to accept during download
MIN_IMAGE_SIZE = 80


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
# ResNet18 embedder (for SVM)
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
    emb = emb / (np.linalg.norm(emb) + 1e-12)  # L2 normalize
    return emb


# =========================
# Common evaluation + Task 5 outputs
# =========================
def exact_match(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    return (y_true.values == y_pred.values).all(axis=1).mean()


def print_multioutput_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    ex = exact_match(y_true, y_pred)
    print(f"Exact Match Accuracy (all {y_true.shape[1]} correct): {ex:.4f}")

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
    """
    Task 5: Analyze misclassifications and confusion patterns from an error CSV.
    """
    df = pd.read_csv(error_csv)
    print("\n" + "=" * 90)
    print(f"TASK 5 ERROR ANALYSIS: {error_csv}")
    print("=" * 90)

    if "exact_ok" in df.columns:
        exact_acc = df["exact_ok"].mean()
        print(f"Exact match (from file): {exact_acc:.4f}")

    # Which label fails most?
    fail_rates = {}
    for col in TARGET_COLS:
        ok_col = f"ok_{col}"
        if ok_col in df.columns:
            fail_rates[col] = 1.0 - df[ok_col].mean()

    print("\nLabel failure rates (higher = harder):")
    for k, v in sorted(fail_rates.items(), key=lambda x: -x[1]):
        print(f"  {k:12s}: {v:.3f}")

    # Top confusions per label
    print("\nTop confusion pairs per label (true -> pred):")
    for col in TARGET_COLS:
        tcol = f"true_{col}"
        pcol = f"pred_{col}"
        if tcol not in df.columns or pcol not in df.columns:
            continue

        conf = Counter()
        wrong = df[df[tcol] != df[pcol]]
        for t, p in zip(wrong[tcol].astype(str), wrong[pcol].astype(str)):
            conf[(t, p)] += 1

        print(f"\n{col}:")
        for (t, p), c in conf.most_common(10):
            print(f"  {t} -> {p}: {c}")


# =========================
# Dataset for CNN
# =========================
class TravelDataset(Dataset):
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

        ys = []
        for col in TARGET_COLS:
            y = self.encoders[col].transform([row[col]])[0]
            ys.append(int(y))

        return x, tuple(ys)


# =========================
# Multi-head CNN model
# =========================
class MultiHeadResNet18(nn.Module):
    def __init__(self, n_weather, n_time, freeze_backbone=False):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # backbone: all layers except final fc
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # -> [B, 512, 1, 1]
        self.feat_dim = 512

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(p=0.5)
        self.head_weather = nn.Linear(self.feat_dim, n_weather)
        self.head_time = nn.Linear(self.feat_dim, n_time)

    def forward(self, x):
        f = self.backbone(x).flatten(1)  # [B, 512]
        f = self.dropout(f)
        return (
            self.head_weather(f),
            self.head_time(f),
        )


def get_train_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=models.ResNet18_Weights.DEFAULT.transforms().mean,
            std=models.ResNet18_Weights.DEFAULT.transforms().std
        ),
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=models.ResNet18_Weights.DEFAULT.transforms().mean,
            std=models.ResNet18_Weights.DEFAULT.transforms().std
        ),
    ])


def train_one_cnn(lr, train_loader, val_loader, encoders, device, freeze_backbone=False, epochs=CNN_EPOCHS):
    n_classes = {col: len(encoders[col].classes_) for col in TARGET_COLS}
    model = MultiHeadResNet18(
        n_weather=n_classes["Weather"],
        n_time=n_classes["Time of Day"],
        freeze_backbone=freeze_backbone
    ).to(device)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_exact = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for x, ys in train_loader:
            x = x.to(device)
            y_weather, y_time = [t.to(device) for t in ys]

            opt.zero_grad()
            o_weather, o_time = model(x)

            loss = (
                ce(o_weather, y_weather) +
                ce(o_time, y_time)
            )
            loss.backward()
            opt.step()

        # validation exact match
        model.eval()
        all_true = {c: [] for c in TARGET_COLS}
        all_pred = {c: [] for c in TARGET_COLS}

        with torch.no_grad():
            for x, ys in val_loader:
                x = x.to(device)
                y_weather, y_time = [t.to(device) for t in ys]
                o_weather, o_time = model(x)

                p_weather = torch.argmax(o_weather, dim=1)
                p_time = torch.argmax(o_time, dim=1)

                all_true["Weather"].extend(y_weather.cpu().numpy().tolist())
                all_true["Time of Day"].extend(y_time.cpu().numpy().tolist())

                all_pred["Weather"].extend(p_weather.cpu().numpy().tolist())
                all_pred["Time of Day"].extend(p_time.cpu().numpy().tolist())

        # compute exact
        ytrue = np.vstack([all_true[c] for c in TARGET_COLS]).T
        ypred = np.vstack([all_pred[c] for c in TARGET_COLS]).T
        val_exact = (ytrue == ypred).all(axis=1).mean()

        print(f"[CNN lr={lr} ep={ep}/{epochs}] val exact: {val_exact:.4f}")

        if val_exact > best_val_exact:
            best_val_exact = val_exact
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # restore best state
    model.load_state_dict(best_state)
    return model, best_val_exact


def cnn_predict(model, loader, encoders, device):
    model.eval()
    all_true = {c: [] for c in TARGET_COLS}
    all_pred = {c: [] for c in TARGET_COLS}

    with torch.no_grad():
        for x, ys in loader:
            x = x.to(device)
            y_weather, y_time = [t.to(device) for t in ys]
            o_weather, o_time = model(x)

            p_weather = torch.argmax(o_weather, dim=1)
            p_time = torch.argmax(o_time, dim=1)

            all_true["Weather"].extend(y_weather.cpu().numpy().tolist())
            all_true["Time of Day"].extend(y_time.cpu().numpy().tolist())

            all_pred["Weather"].extend(p_weather.cpu().numpy().tolist())
            all_pred["Time of Day"].extend(p_time.cpu().numpy().tolist())

    # decode to original strings
    y_true_df = pd.DataFrame({
        col: encoders[col].inverse_transform(np.array(all_true[col], dtype=int))
        for col in TARGET_COLS
    })
    y_pred_df = pd.DataFrame({
        col: encoders[col].inverse_transform(np.array(all_pred[col], dtype=int))
        for col in TARGET_COLS
    })
    return y_true_df, y_pred_df


# =========================
# MAIN: Task 4 + Task 5
# =========================
def main():
    seed_everything()

    df = pd.read_csv(CSV_PATH)

    # Cache/download images (reuses cache on reruns)
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

    df = df.loc[keep_mask].copy()
    df["local_path"] = [p for p in local_paths if p is not None]
    df = df.reset_index(drop=True)

    print(f"Rows kept: {len(df)}")

    # Split: Train / Test (same style as Task 3)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

    # Further split Train -> Train/Val for tuning
    train_df2, val_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

    # =========================================================
    # MODEL A: SVM on ResNet18 embeddings
    # =========================================================
    print("\n" + "=" * 90)
    print("TASK 4 - MODEL A: Linear SVM (MultiOutput) on ResNet18 embeddings")
    print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder, preprocess = build_resnet18_embedder(device=device)

    print("Extracting embeddings for SVM...")
    X_train = np.vstack([image_to_embedding(p, embedder, preprocess, device) for p in tqdm(train_df2["local_path"])])
    X_val   = np.vstack([image_to_embedding(p, embedder, preprocess, device) for p in tqdm(val_df["local_path"])])
    X_test  = np.vstack([image_to_embedding(p, embedder, preprocess, device) for p in tqdm(test_df["local_path"])])

    y_train = train_df2[TARGET_COLS].copy()
    y_val   = val_df[TARGET_COLS].copy()
    y_test  = test_df[TARGET_COLS].copy()

    best_C = None
    best_val_exact = -1.0
    best_svm = None

    for C in SVM_C_VALUES:
        svm = MultiOutputClassifier(LinearSVC(C=C, max_iter=5000, dual="auto", class_weight='balanced'))
        svm.fit(X_train, y_train)
        val_pred = pd.DataFrame(svm.predict(X_val), columns=TARGET_COLS)
        val_exact = print_multioutput_metrics(y_val, val_pred, title=f"SVM (C={C}) - VALIDATION")
        if val_exact > best_val_exact:
            best_val_exact = val_exact
            best_C = C
            best_svm = svm

    # Test with best C
    test_pred_svm = pd.DataFrame(best_svm.predict(X_test), columns=TARGET_COLS)
    print_multioutput_metrics(y_test, test_pred_svm, title=f"SVM BEST (C={best_C}) - TEST")

    # Task 5 output for SVM
    svm_err_csv = f"task4_svm_errors_C{best_C}.csv"
    save_error_table(
        df_meta=test_df[["Image URL", "Description", "Country", "Activity"]].copy(),
        y_true=y_test,
        y_pred=test_pred_svm,
        out_csv=svm_err_csv
    )
    task5_analyze_errors(svm_err_csv)

    # =========================================================
    # MODEL B: Fine-tuned Multi-Head CNN (ResNet18)
    # =========================================================
    print("\n" + "=" * 90)
    print("TASK 4 - MODEL B: Fine-tuned ResNet18 Multi-Head CNN")
    print("=" * 90)

    # Label encoders for CNN
    encoders = {}
    for col in TARGET_COLS:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))  # fit on full training split (train_df)
        encoders[col] = le

        encoders[col] = le
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_ds = TravelDataset(train_df2, encoders, train_transform)
    val_ds   = TravelDataset(val_df, encoders, val_transform)
    test_ds  = TravelDataset(test_df, encoders, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device_t = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CNN training device: {device_t}")

    best_lr = None
    best_val_exact = -1.0
    best_cnn = None

    for lr in CNN_LR_VALUES:
        model, val_exact = train_one_cnn(
            lr=lr,
            train_loader=train_loader,
            val_loader=val_loader,
            encoders=encoders,
            device=device_t,
            freeze_backbone=False,
            epochs=CNN_EPOCHS
        )
        if val_exact > best_val_exact:
            best_val_exact = val_exact
            best_lr = lr
            best_cnn = model

    # Test best CNN
    y_true_cnn, y_pred_cnn = cnn_predict(best_cnn, test_loader, encoders, device_t)
    print_multioutput_metrics(y_true_cnn, y_pred_cnn, title=f"CNN BEST (lr={best_lr}) - TEST")

    # Task 5 output for CNN
    cnn_err_csv = f"task4_cnn_errors_lr{best_lr}.csv"
    save_error_table(
        df_meta=test_df[["Image URL", "Description", "Country", "Activity"]].copy(),
        y_true=y_true_cnn,
        y_pred=y_pred_cnn,
        out_csv=cnn_err_csv
    )
    task5_analyze_errors(cnn_err_csv)

    print("\nDONE ✅")
    print(f"- SVM errors: {svm_err_csv}")
    print(f"- CNN errors: {cnn_err_csv}")


if __name__ == "__main__":
    main()
