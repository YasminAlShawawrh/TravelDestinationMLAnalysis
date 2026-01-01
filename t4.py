"""
Task 4: Advanced Machine Learning Models
Predict Weather, Time of Day, and Season using:
- Random Forest
- Support Vector Machine (SVM)
- CNN (Convolutional Neural Network) - processes images directly
ENCS5341 - Assignment 3

FIXES INCLUDED:
✅ max_batches_per_epoch can be None (train on ALL batches) without crashing
✅ CNN evaluation uses FULL urls list + indices (no out-of-range bugs)
✅ Confusion-matrix plotting adapts to number of models available
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch imports for CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import requests
from io import BytesIO
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Import feature extractor
from image_feature_extractor import ImageFeatureExtractor


# ----------------------------
# Dataset
# ----------------------------
class ImageDataset(Dataset):
    """Dataset class for loading images from URLs"""

    def __init__(self, urls, labels_dict, transform=None):
        self.urls = urls
        self.labels_dict = labels_dict
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, timeout=8, headers=headers, verify=False, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), color='black')

        img = self.transform(img)

        # Return image and labels for all targets
        labels = {key: val[idx] for key, val in self.labels_dict.items()}
        return img, labels


# ----------------------------
# CNN Model
# ----------------------------
class CNNModel(nn.Module):
    """CNN model for multi-target classification"""

    def __init__(self, num_classes_dict):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.heads = nn.ModuleDict()
        for target_name, num_classes in num_classes_dict.items():
            self.heads[target_name] = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        outputs = {}
        for target_name in self.heads.keys():
            outputs[target_name] = self.heads[target_name](x)
        return outputs


# ----------------------------
# Advanced Models Wrapper
# ----------------------------
class AdvancedMLModels:
    """Advanced ML models for multi-target classification"""

    def __init__(self):
        self.models = {
            'Random Forest': {},
            'SVM': {},
            'CNN': None
        }
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_columns = ['Weather', 'Time of Day', 'Season']
        self.cnn_device = None

    def prepare_data(self, df, features, remove_unknown=True, min_samples_per_class=3):
        if len(features) != len(df):
            raise ValueError(
                f"Feature array size ({len(features)}) doesn't match dataframe size ({len(df)}). "
                f"Please re-extract features from cleaned_data.csv."
            )

        # Remove Unknown rows (if your preprocessing uses Unknown)
        if remove_unknown:
            mask = ~(df[self.target_columns].isin(['Unknown']).any(axis=1))
            df = df[mask].reset_index(drop=True)
            features = features[mask]

        # Remove rare classes
        for col in self.target_columns:
            value_counts = df[col].value_counts()
            rare_classes = value_counts[value_counts < min_samples_per_class].index.tolist()
            if rare_classes:
                mask = ~df[col].isin(rare_classes)
                df = df[mask].reset_index(drop=True)
                features = features[mask]

        X = features

        y_dict = {}
        for col in self.target_columns:
            le = LabelEncoder()
            y_dict[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        return df, X, y_dict  # return filtered df too

    def train_models(self, X_train, y_train_dict):
        X_train_scaled = self.scaler.fit_transform(X_train)

        for col in self.target_columns:
            print(f"\n{'='*70}")
            print(f"Training models for {col}...")
            print('='*70)

            print(f"\n1. Training Random Forest for {col}...")
            rf = RandomForestClassifier(
                n_estimators=150,
                max_depth=25,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train_dict[col])
            self.models['Random Forest'][col] = rf
            print("   [OK] Random Forest trained")

            print(f"\n2. Training SVM for {col}...")
            svm = SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
            svm.fit(X_train_scaled, y_train_dict[col])
            self.models['SVM'][col] = svm
            print("   [OK] SVM trained")

    def train_cnn(
        self,
        urls_full,
        y_train_dict,
        y_test_dict,
        train_indices,
        test_indices,
        epochs=15,
        batch_size=16,
        learning_rate=0.001,
        max_batches_per_epoch=None
    ):
        print(f"\n{'='*70}")
        print("Training CNN (Convolutional Neural Network) on images...")
        print("=" * 70)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_device = device
        print(f"Using device: {device}")

        num_classes_dict = {col: len(self.label_encoders[col].classes_) for col in self.target_columns}
        model = CNNModel(num_classes_dict).to(device)

        # Build train/test urls using indices from FULL urls list
        train_urls = [urls_full[i] for i in train_indices]
        test_urls = [urls_full[i] for i in test_indices]

        # y_train_dict[col] aligned with train_idx length
        train_labels = {col: y_train_dict[col].tolist() for col in self.target_columns}
        test_labels = {col: y_test_dict[col].tolist() for col in self.target_columns}

        train_dataset = ImageDataset(train_urls, train_labels)
        test_dataset = ImageDataset(test_urls, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        _ = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if max_batches_per_epoch is None:
            print(f"\nTraining for {epochs} epochs (using ALL batches per epoch)...")
        else:
            print(f"\nTraining for {epochs} epochs (max {max_batches_per_epoch} batches per epoch)...")

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            batches_seen = 0

            for _, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                outputs = model(images)

                loss = 0
                for col in self.target_columns:
                    target_labels = torch.tensor(labels[col], dtype=torch.long).to(device)
                    loss = loss + criterion(outputs[col], target_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batches_seen += 1

                # ✅ FIX: Only compare if not None
                if max_batches_per_epoch is not None and batches_seen >= max_batches_per_epoch:
                    break

            avg_loss = running_loss / max(1, batches_seen)
            print(f"  Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.4f}")

        self.models['CNN'] = model
        print("\n[OK] CNN trained successfully")

    def predict(self, X, model_name, urls=None, indices=None):
        if model_name == 'CNN':
            if urls is None:
                raise ValueError("urls are required for CNN prediction")

            # If indices are provided and urls is FULL list -> subset by indices
            # If urls is already a subset aligned with desired prediction length -> use directly
            if indices is not None and len(urls) > len(indices) and (max(indices) < len(urls)):
                use_urls = [urls[i] for i in indices]
            else:
                use_urls = urls  # already aligned

            predictions = {}
            model = self.models['CNN']
            model.eval()

            dummy_labels = {col: np.zeros(len(use_urls)) for col in self.target_columns}
            dataset = ImageDataset(use_urls, dummy_labels)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            all_preds = {col: [] for col in self.target_columns}

            with torch.no_grad():
                for images, _ in dataloader:
                    images = images.to(self.cnn_device)
                    outputs = model(images)
                    for col in self.target_columns:
                        preds = torch.argmax(outputs[col], dim=1).cpu().numpy()
                        all_preds[col].extend(preds)

            for col in self.target_columns:
                pred_encoded = np.array(all_preds[col])
                pred_labels = self.label_encoders[col].inverse_transform(pred_encoded)
                predictions[col] = pred_labels

            return predictions

        # Traditional models
        X_scaled = self.scaler.transform(X)
        predictions = {}
        for col in self.target_columns:
            model = self.models[model_name][col]
            pred_encoded = model.predict(X_scaled)
            pred_labels = self.label_encoders[col].inverse_transform(pred_encoded)
            predictions[col] = pred_labels
        return predictions

    def evaluate(self, X_test, y_test_dict, urls_full=None, test_indices=None):
        all_results = {}

        for model_name in self.models.keys():
            if model_name == 'CNN' and self.models['CNN'] is None:
                continue

            print(f"\n{'='*70}")
            print(f"Evaluating {model_name}...")
            print('='*70)

            if model_name == 'CNN':
                # IMPORTANT: pass FULL urls list + indices
                predictions = self.predict(X_test, model_name, urls=urls_full, indices=test_indices)
            else:
                predictions = self.predict(X_test, model_name)

            results = {}
            for col in self.target_columns:
                y_true = y_test_dict[col]
                y_pred_encoded = self.label_encoders[col].transform(predictions[col])

                accuracy = accuracy_score(y_true, y_pred_encoded)
                f1 = f1_score(y_true, y_pred_encoded, average='weighted')

                results[col] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': predictions[col],
                    'true_labels': self.label_encoders[col].inverse_transform(y_true)
                }

                print(f"\n{col} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

            all_results[model_name] = results

        return all_results

    def plot_comparison(self, all_results, output_dir='task4_outputs'):
        os.makedirs(output_dir, exist_ok=True)

        model_names = [m for m in self.models.keys() if (m in all_results)]
        if len(model_names) == 0:
            print("[WARNING] No results to plot.")
            return

        # Accuracy bar chart (targets = 3)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Model Comparison - Accuracy Scores', fontsize=16, fontweight='bold')

        for idx, col in enumerate(self.target_columns):
            accuracies = [all_results[m][col]['accuracy'] for m in model_names]
            bars = axes[idx].bar(model_names, accuracies, alpha=0.7)
            axes[idx].set_ylabel('Accuracy', fontsize=12)
            axes[idx].set_title(f'{col}', fontsize=14, fontweight='bold')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)

            for bar in bars:
                h = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width() / 2., h, f'{h:.3f}',
                               ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved comparison plot to {output_dir}/model_comparison.png")
        plt.close()

        # Confusion matrices: number of columns matches number of models
        for col in self.target_columns:
            n_models = len(model_names)
            fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
            if n_models == 1:
                axes = [axes]

            fig.suptitle(f'Confusion Matrices - {col}', fontsize=16, fontweight='bold')

            for i, model_name in enumerate(model_names):
                y_true = all_results[model_name][col]['true_labels']
                y_pred = all_results[model_name][col]['predictions']

                y_true_encoded = self.label_encoders[col].transform(y_true)
                y_pred_encoded = self.label_encoders[col].transform(y_pred)

                cm = confusion_matrix(y_true_encoded, y_pred_encoded)
                labels = self.label_encoders[col].classes_

                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[i], cbar_kws={'label': 'Count'}
                )
                axes[i].set_title(f'{model_name}\nAcc: {all_results[model_name][col]["accuracy"]:.4f}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
                plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
                plt.setp(axes[i].get_yticklabels(), rotation=0)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/confusion_matrices_{col.replace(" ", "_")}.png',
                        dpi=300, bbox_inches='tight')
            print(f"[OK] Saved confusion matrices for {col}")
            plt.close()

    def generate_reports(self, all_results, y_test_dict, output_dir='task4_outputs'):
        os.makedirs(output_dir, exist_ok=True)

        report_file = f'{output_dir}/detailed_reports.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("DETAILED CLASSIFICATION REPORTS - ADVANCED MODELS\n")
            f.write("=" * 70 + "\n\n")

            for model_name in all_results.keys():
                f.write(f"\n{'='*70}\n")
                f.write(f"{model_name.upper()}\n")
                f.write('=' * 70 + "\n\n")

                for col in self.target_columns:
                    f.write(f"\n{col}:\n")
                    f.write("-" * 70 + "\n")

                    y_true = y_test_dict[col]
                    y_pred = all_results[model_name][col]['predictions']
                    y_pred_encoded = self.label_encoders[col].transform(y_pred)

                    report = classification_report(
                        y_true, y_pred_encoded,
                        target_names=self.label_encoders[col].classes_,
                        output_dict=False
                    )
                    f.write(report + "\n")

        print(f"[OK] Saved detailed reports to {report_file}")


def main():
    print("=" * 70)
    print("TASK 4: ADVANCED MACHINE LEARNING MODELS")
    print("Predicting: Weather, Time of Day, Season")
    print("Models: Random Forest, SVM, CNN")
    print("=" * 70)

    print("\n1. Loading cleaned data...")
    df = pd.read_csv('cleaned_data.csv')
    print(f"   [OK] Loaded {len(df)} samples")

    print("\n2. Loading image features...")
    features = None
    try:
        loaded_features = np.load('image_features.npy')
        print(f"   [OK] Loaded features: {loaded_features.shape}")

        if loaded_features.shape[0] == len(df):
            features = loaded_features
            print(f"   [OK] Feature size matches dataframe size ({len(df)} rows)")
        else:
            print(f"   [WARNING] Feature size ({loaded_features.shape[0]}) != dataframe size ({len(df)})")
            print("   [INFO] Re-extracting features...")
            extractor = ImageFeatureExtractor()
            features, _ = extractor.extract_features_from_dataframe(df, save_path='image_features.npy', verbose=True)
            print(f"   [OK] Extracted features: {features.shape}")
    except FileNotFoundError:
        print("   Features not found. Extracting from images...")
        extractor = ImageFeatureExtractor()
        features, _ = extractor.extract_features_from_dataframe(df, save_path='image_features.npy', verbose=True)
        print(f"   [OK] Extracted features: {features.shape}")

    if features is None:
        raise ValueError("Failed to load or extract features")

    print("\n3. Initializing advanced models...")
    models = AdvancedMLModels()

    print("\n4. Preparing data...")
    filtered_df, X, y_dict = models.prepare_data(df, features, remove_unknown=True, min_samples_per_class=3)
    print(f"   [OK] Prepared {len(X)} samples")

    # URLs must match filtered_df
    if 'Image URL' not in filtered_df.columns:
        raise ValueError("Column 'Image URL' not found in cleaned_data.csv")
    urls_full = filtered_df['Image URL'].tolist()
    print(f"   [OK] URLs aligned: {len(urls_full)} URLs")

    print("\n5. Splitting data...")
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]

    y_train_dict, y_test_dict = {}, {}
    for col in models.target_columns:
        y_train_dict[col] = y_dict[col][train_idx]
        y_test_dict[col] = y_dict[col][test_idx]

    print(f"   [OK] Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n6. Training traditional models (Random Forest, SVM)...")
    models.train_models(X_train, y_train_dict)

    print("\n7. Training CNN on images (PROPER TRAINING SETTINGS)...")
    models.train_cnn(
        urls_full=urls_full,
        y_train_dict=y_train_dict,
        y_test_dict=y_test_dict,
        train_indices=train_idx,
        test_indices=test_idx,
        epochs=15,
        batch_size=16,
        learning_rate=0.001,
        max_batches_per_epoch=None  # ✅ train on ALL batches
    )

    print("\n8. Evaluating all models...")
    all_results = models.evaluate(
        X_test,
        y_test_dict,
        urls_full=urls_full,
        test_indices=test_idx
    )

    print("\n9. Generating detailed reports...")
    models.generate_reports(all_results, y_test_dict)

    print("\n10. Generating visualizations...")
    models.plot_comparison(all_results)

    print("\n" + "=" * 70)
    print("TASK 4 COMPLETE!")
    print("=" * 70)

    print("\nSummary of Results:")
    print("\n" + "-" * 70)
    print(f"{'Model':<20} {'Weather':<12} {'Time of Day':<15} {'Season':<12}")
    print("-" * 70)
    for model_name in all_results.keys():
        accs = [all_results[model_name][col]['accuracy'] for col in models.target_columns]
        print(f"{model_name:<20} {accs[0]:<12.4f} {accs[1]:<15.4f} {accs[2]:<12.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
