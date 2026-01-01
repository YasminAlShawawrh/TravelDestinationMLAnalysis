"""
Task 3: Baseline KNN Model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import feature extractor
from image_feature_extractor import ImageFeatureExtractor

class BaselineKNNModel:
    
    def __init__(self, n_neighbors=1, metric='minkowski', p=2):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.models = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_columns = ['Weather', 'Time of Day', 'Season']
        
    def prepare_data(self, df, features, remove_unknown=True, min_samples_per_class=3):

        if len(features) != len(df):
            raise ValueError(f"Feature array size ({len(features)}) doesn't match dataframe size ({len(df)}). "
                           f"Please re-extract features from cleaned_data.csv.")

        if remove_unknown:
            mask = ~(df[self.target_columns].isin(['Unknown']).any(axis=1))
            df = df[mask].reset_index(drop=True)
            features = features[mask]
        
        for col in self.target_columns:
            value_counts = df[col].value_counts()
            rare_classes = value_counts[value_counts < min_samples_per_class].index.tolist()
            if rare_classes:
                mask = ~df[col].isin(rare_classes)
                df = df[mask].reset_index(drop=True)
                features = features[mask]
        
        X = features
        
        # Encode labels
        y_dict = {}
        for col in self.target_columns:
            le = LabelEncoder()
            y_dict[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        return X, y_dict
    
    def train(self, X_train, y_train_dict):

        X_train_scaled = self.scaler.fit_transform(X_train)
        
        for col in self.target_columns:
            print(f"\nTraining KNN (k={self.n_neighbors}) for {col}...")
            knn = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights='uniform', 
                metric=self.metric,
                p=self.p 
            )
            knn.fit(X_train_scaled, y_train_dict[col])
            self.models[col] = knn
            print(f"[OK] Trained KNN (k={self.n_neighbors}) for {col}")
    
    def predict(self, X):

        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for col in self.target_columns:
            pred_encoded = self.models[col].predict(X_scaled)
            pred_labels = self.label_encoders[col].inverse_transform(pred_encoded)
            predictions[col] = pred_labels
        
        return predictions
    
    def evaluate(self, X_test, y_test_dict):

        predictions = self.predict(X_test)
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
        
        return results
    
    def plot_confusion_matrices(self, y_test_dict, results, output_dir='task3_outputs', k_value=1):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Confusion Matrices - Baseline KNN Model (k={k_value})', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(self.target_columns):
            y_true = y_test_dict[col]
            predictions = results[col]['predictions']
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            y_pred_encoded = self.label_encoders[col].transform(predictions)
            
            cm = confusion_matrix(y_true, y_pred_encoded)
            labels = self.label_encoders[col].classes_
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[idx], cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{col}\nAccuracy: {accuracy_score(y_true, y_pred_encoded):.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            plt.setp(axes[idx].get_xticklabels(), rotation=45, ha='right')
            plt.setp(axes[idx].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrices_knn_k{k_value}.png', dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved confusion matrices to {output_dir}/confusion_matrices_knn_k{k_value}.png")
        plt.close()
    
    def compare_k_values(self, k1_results, k3_results, output_dir='task3_outputs'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Baseline KNN Comparison: k=1 vs k=3', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(self.target_columns):
            k1_acc = k1_results[col]['accuracy']
            k3_acc = k3_results[col]['accuracy']
            k1_f1 = k1_results[col]['f1_score']
            k3_f1 = k3_results[col]['f1_score']
            
            x = np.arange(2)
            width = 0.35
            
            axes[idx].bar(x - width/2, [k1_acc, k1_f1], width, label='k=1', color='#3498db', alpha=0.7)
            axes[idx].bar(x + width/2, [k3_acc, k3_f1], width, label='k=3', color='#e74c3c', alpha=0.7)
            
            axes[idx].set_ylabel('Score', fontsize=12)
            axes[idx].set_title(f'{col}', fontsize=14, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(['Accuracy', 'F1-Score'])
            axes[idx].legend()
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].set_ylim([0, 1])
            
            axes[idx].text(x[0] - width/2, k1_acc + 0.02, f'{k1_acc:.3f}', 
                          ha='center', va='bottom', fontsize=9)
            axes[idx].text(x[0] + width/2, k3_acc + 0.02, f'{k3_acc:.3f}', 
                          ha='center', va='bottom', fontsize=9)
            axes[idx].text(x[1] - width/2, k1_f1 + 0.02, f'{k1_f1:.3f}', 
                          ha='center', va='bottom', fontsize=9)
            axes[idx].text(x[1] + width/2, k3_f1 + 0.02, f'{k3_f1:.3f}', 
                          ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/knn_comparison_k1_vs_k3.png', dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved comparison plot to {output_dir}/knn_comparison_k1_vs_k3.png")
        plt.close()


def main():
    print("=" * 70)
    print("TASK 3: BASELINE KNN MODEL")
    print("Predicting: Weather, Time of Day, Season")
    print("Evaluating k=1 and k=3 as baseline models")
    print("Distance Metric: Euclidean (Minkowski with p=2)")
    print("=" * 70)

    print("\n1. Loading cleaned data...")
    df = pd.read_csv('cleaned_data.csv')
    print(f"   [OK] Loaded {len(df)} samples")
    
    print("\n2. Loading/extracting image features...")
    features = None
    try:
        loaded_features = np.load('image_features.npy')
        print(f"   [OK] Loaded features from image_features.npy")
        print(f"   Feature shape: {loaded_features.shape}")
        
        if loaded_features.shape[0] == len(df):
            features = loaded_features
            print(f"   [OK] Feature size matches dataframe size ({len(df)} rows)")
        else:
            print(f"   [WARNING] Feature size ({loaded_features.shape[0]}) doesn't match dataframe size ({len(df)})")
            print(f"   [INFO] Re-extracting features from cleaned_data.csv...")
            extractor = ImageFeatureExtractor()
            features, failed_indices = extractor.extract_features_from_dataframe(
                df, save_path='image_features.npy', verbose=True
            )
            print(f"   [OK] Extracted features: {features.shape}")
    except FileNotFoundError:
        print("   Features not found. Extracting from images...")
        extractor = ImageFeatureExtractor()
        features, failed_indices = extractor.extract_features_from_dataframe(
            df, save_path='image_features.npy', verbose=True
        )
        print(f"   [OK] Extracted features: {features.shape}")
    
    if features is None:
        raise ValueError("Failed to load or extract features")
    
    print("\n3. Preparing data...")
    temp_model = BaselineKNNModel(n_neighbors=1)
    X, y_dict = temp_model.prepare_data(df, features, remove_unknown=True, min_samples_per_class=3)
    print(f"   [OK] Prepared {len(X)} samples after filtering")
    
    print("\n4. Splitting data into train/test sets...")
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    y_train_dict = {}
    y_test_dict = {}
    for col in temp_model.target_columns:
        y_train_dict[col] = y_dict[col][train_idx]
        y_test_dict[col] = y_dict[col][test_idx]
    
    print(f"   [OK] Train set: {len(X_train)} samples")
    print(f"   [OK] Test set: {len(X_test)} samples")
    
    label_encoders = temp_model.label_encoders
    # EVALUATE K=1 BASELINE
    print("\n" + "=" * 70)
    print("EVALUATING BASELINE KNN WITH k=1")
    print("=" * 70)
    
    model_k1 = BaselineKNNModel(n_neighbors=1, metric='minkowski', p=2)
    model_k1.label_encoders = label_encoders  # Reuse encoders
    
    print("\n5. Training KNN models with k=1...")
    model_k1.train(X_train, y_train_dict)
    
    print("\n6. Evaluating k=1 model performance...")
    results_k1 = model_k1.evaluate(X_test, y_test_dict)
    
    print("\n7. Performing cross-validation for k=1...")
    X_scaled_k1 = model_k1.scaler.fit_transform(X_train)
    cv_scores_k1 = {}
    for col in model_k1.target_columns:
        cv_scores = cross_val_score(model_k1.models[col], X_scaled_k1, y_train_dict[col], 
                                    cv=5, scoring='accuracy')
        cv_scores_k1[col] = cv_scores
        print(f"\n{col} (k=1) - Cross-validation scores: {cv_scores}")
        print(f"{col} (k=1) - Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # EVALUATE K=3 BASELINE
    print("\n" + "=" * 70)
    print("EVALUATING BASELINE KNN WITH k=3")
    print("=" * 70)
    
    model_k3 = BaselineKNNModel(n_neighbors=3, metric='minkowski', p=2)
    model_k3.label_encoders = label_encoders  # Reuse encoders
    
    print("\n8. Training KNN models with k=3...")
    model_k3.train(X_train, y_train_dict)
    
    print("\n9. Evaluating k=3 model performance...")
    results_k3 = model_k3.evaluate(X_test, y_test_dict)
    
    print("\n10. Performing cross-validation for k=3...")
    X_scaled_k3 = model_k3.scaler.fit_transform(X_train)
    cv_scores_k3 = {}
    for col in model_k3.target_columns:
        cv_scores = cross_val_score(model_k3.models[col], X_scaled_k3, y_train_dict[col], 
                                    cv=5, scoring='accuracy')
        cv_scores_k3[col] = cv_scores
        print(f"\n{col} (k=3) - Cross-validation scores: {cv_scores}")
        print(f"{col} (k=3) - Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # DETAILED REPORTS
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION REPORTS")
    print("=" * 70)
    
    print("\n11. Generating detailed classification reports for k=1...")
    for col in model_k1.target_columns:
        print(f"\n{'='*70}")
        print(f"Classification Report - {col} (k=1)")
        print('='*70)
        y_true = y_test_dict[col]
        y_pred_encoded = model_k1.label_encoders[col].transform(results_k1[col]['predictions'])
        print(classification_report(y_true, y_pred_encoded, 
                                   target_names=model_k1.label_encoders[col].classes_))
    
    print("\n12. Generating detailed classification reports for k=3...")
    for col in model_k3.target_columns:
        print(f"\n{'='*70}")
        print(f"Classification Report - {col} (k=3)")
        print('='*70)
        y_true = y_test_dict[col]
        y_pred_encoded = model_k3.label_encoders[col].transform(results_k3[col]['predictions'])
        print(classification_report(y_true, y_pred_encoded, 
                                   target_names=model_k3.label_encoders[col].classes_))
    
    # VISUALIZATIONS
    print("\n13. Generating visualizations...")
    model_k1.plot_confusion_matrices(y_test_dict, results_k1, k_value=1)
    model_k3.plot_confusion_matrices(y_test_dict, results_k3, k_value=3)
    model_k1.compare_k_values(results_k1, results_k3)
    
    # SUMMARY COMPARISON
    print("\n" + "=" * 70)
    print("TASK 3 COMPLETE - BASELINE KNN COMPARISON")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("PERFORMANCE SUMMARY: k=1 vs k=3")
    print("-" * 70)
    print(f"\n{'Target':<20} {'Metric':<15} {'k=1':<12} {'k=3':<12} {'Best':<10}")
    print("-" * 70)
    
    for col in model_k1.target_columns:
        k1_acc = results_k1[col]['accuracy']
        k3_acc = results_k3[col]['accuracy']
        k1_f1 = results_k1[col]['f1_score']
        k3_f1 = results_k3[col]['f1_score']
        
        best_acc = "k=1" if k1_acc >= k3_acc else "k=3"
        best_f1 = "k=1" if k1_f1 >= k3_f1 else "k=3"
        
        print(f"{col:<20} {'Accuracy':<15} {k1_acc:<12.4f} {k3_acc:<12.4f} {best_acc:<10}")
        print(f"{'':<20} {'F1-Score':<15} {k1_f1:<12.4f} {k3_f1:<12.4f} {best_f1:<10}")
        print()
    
    print("-" * 70)
    print("\nCross-Validation Summary:")
    print("-" * 70)
    for col in model_k1.target_columns:
        k1_cv_mean = cv_scores_k1[col].mean()
        k3_cv_mean = cv_scores_k3[col].mean()
        best_cv = "k=1" if k1_cv_mean >= k3_cv_mean else "k=3"
        print(f"{col}: k=1 CV: {k1_cv_mean:.4f}, k=3 CV: {k3_cv_mean:.4f} (Best: {best_cv})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
