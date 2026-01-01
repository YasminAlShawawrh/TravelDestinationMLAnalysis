"""
Task 5: Performance Analysis [20 points]
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns
# Helper functions
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Count"}
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
def most_confused_pairs(cm, labels, top_k=5):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            count = cm[i, j]
            if count > 0:
                pairs.append((labels[i], labels[j], int(count)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]
def per_class_recall(cm, labels):
    recalls = []
    for i, lbl in enumerate(labels):
        row_sum = cm[i, :].sum()
        tp = cm[i, i]
        rec = (tp / row_sum) if row_sum > 0 else 0.0
        recalls.append((lbl, rec, int(tp), int(row_sum)))
    df = pd.DataFrame(recalls, columns=["Class", "Recall", "TP", "Support"])
    df = df.sort_values("Recall", ascending=True).reset_index(drop=True)
    return df
    
def main():
    print("=" * 70)
    print("TASK 5: PERFORMANCE ANALYSIS (Best Model Error Analysis)")
    print("Best Model: Random Forest on image embeddings")
    print("=" * 70)

    output_dir = "task5_outputs"
    ensure_dir(output_dir)

    # 1) Load data
    print("\n1. Loading cleaned data and features...")
    df = pd.read_csv("cleaned_data.csv")
    features = np.load("image_features.npy")

    if len(df) != features.shape[0]:
        raise ValueError(
            f"Mismatch: cleaned_data.csv has {len(df)} rows but features has {features.shape[0]} rows."
        )

    print(f"   [OK] Samples: {len(df)}")
    print(f"   [OK] Features shape: {features.shape}")

    target_columns = ["Weather", "Time of Day", "Season"]

    url_col = None
    for candidate in ["Image URL", "ImageURL", "url", "URL"]:
        if candidate in df.columns:
            url_col = candidate
            break

    desc_col = None
    for candidate in ["Description", "description", "Text", "Caption"]:
        if candidate in df.columns:
            desc_col = candidate
            break

    if url_col is None:
        print("   [WARNING] No Image URL column found. Misclassified CSV will not contain URLs.")
    else:
        print(f"   [OK] URL column: {url_col}")

    if desc_col is not None:
        print(f"   [OK] Description column: {desc_col}")

    # 2) Encode labels
    print("\n2. Encoding labels...")
    label_encoders = {}
    y_dict = {}
    for col in target_columns:
        le = LabelEncoder()
        y_dict[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   [OK] {col}: {len(le.classes_)} classes")

    # 3) Train/Test split (same style as Tasks 3/4)
    print("\n3. Splitting into train/test...")
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train = features[train_idx]
    X_test = features[test_idx]

    # 4) Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Train Random Forest per target (best model)
    print("\n4. Training Random Forest (best model) per target...")
    rf_models = {}
    for col in target_columns:
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_dict[col][train_idx])
        rf_models[col] = rf
        print(f"   [OK] Trained RF for {col}")

    # 6) Evaluate + Error analysis
    print("\n5. Evaluating and analyzing errors...")
    summary_rows = []

    for col in target_columns:
        print(f"\n{'='*70}")
        print(f"ANALYSIS FOR TARGET: {col}")
        print(f"{'='*70}")

        model = rf_models[col]
        y_true = y_dict[col][test_idx]
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted")

        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted F1: {f1w:.4f}")

        labels = label_encoders[col].classes_
        cm = confusion_matrix(y_true, y_pred)

        cm_path = os.path.join(output_dir, f"confusion_matrix_RF_{col.replace(' ', '_')}.png")
        plot_confusion_matrix(cm, labels, f"Random Forest Confusion Matrix - {col}", cm_path)
        print(f"[OK] Saved confusion matrix: {cm_path}")

        report = classification_report(y_true, y_pred, target_names=labels)
        rep_path = os.path.join(output_dir, f"classification_report_RF_{col.replace(' ', '_')}.txt")
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[OK] Saved report: {rep_path}")

        recall_df = per_class_recall(cm, labels)
        recall_csv = os.path.join(output_dir, f"per_class_recall_RF_{col.replace(' ', '_')}.csv")
        recall_df.to_csv(recall_csv, index=False)
        print(f"[OK] Saved per-class recall table: {recall_csv}")
        print("\nLowest recall classes (hardest):")
        print(recall_df.head(5).to_string(index=False))

        confused = most_confused_pairs(cm, labels, top_k=8)
        confused_path = os.path.join(output_dir, f"most_confused_pairs_RF_{col.replace(' ', '_')}.txt")
        with open(confused_path, "w", encoding="utf-8") as f:
            f.write(f"Most confused (Actual -> Predicted) pairs for {col}\n")
            f.write("-" * 60 + "\n")
            for a, p, c in confused:
                f.write(f"{a} -> {p}: {c}\n")
        print(f"[OK] Saved confused pairs: {confused_path}")
        print("\nMost confused pairs:")
        for a, p, c in confused[:5]:
            print(f"  {a} -> {p}: {c}")

        wrong_mask = (y_true != y_pred)
        wrong_test_indices = test_idx[wrong_mask]

        mis_df = pd.DataFrame({
            "Index": wrong_test_indices,
            "Target": col,
            "True_Label": label_encoders[col].inverse_transform(y_true[wrong_mask]),
            "Pred_Label": label_encoders[col].inverse_transform(y_pred[wrong_mask]),
        })

        if url_col is not None:
            mis_df["Image_URL"] = df.loc[wrong_test_indices, url_col].values

        if desc_col is not None:
            mis_df["Description"] = df.loc[wrong_test_indices, desc_col].values

        mis_csv = os.path.join(output_dir, f"misclassified_examples_RF_{col.replace(' ', '_')}.csv")
        mis_df.to_csv(mis_csv, index=False)
        print(f"[OK] Saved misclassified examples CSV: {mis_csv}")
        print(f"Misclassifications count: {len(mis_df)}")

        summary_rows.append([col, acc, f1w, len(mis_df)])

    # 7) Save overall summary
    summary_df = pd.DataFrame(summary_rows, columns=["Target", "Accuracy", "Weighted_F1", "Num_Misclassified"])
    summary_path = os.path.join(output_dir, "task5_summary_RF.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("TASK 5 COMPLETE ")
    print("=" * 70)
    print("\nSaved outputs in:", output_dir)
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
