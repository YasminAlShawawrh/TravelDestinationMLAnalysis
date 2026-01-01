"""
Task 5: Hyperparameter Tuning and Model Optimization
Perform grid search and random search for hyperparameter optimization
Compare tuned models with baseline models
ENCS5341 - Assignment 3
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import feature extractor
from image_feature_extractor import ImageFeatureExtractor


class HyperparameterTuning:
    """Hyperparameter tuning for all models"""
    
    def __init__(self):
        """Initialize tuner"""
        self.best_models = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_columns = ['Weather', 'Time of Day', 'Season']
        self.tuning_results = {}
        
    def prepare_data(self, df, features, remove_unknown=True, min_samples_per_class=3):
        """Prepare data for training"""
        # Ensure features and dataframe have the same number of rows
        if len(features) != len(df):
            raise ValueError(f"Feature array size ({len(features)}) doesn't match dataframe size ({len(df)}). "
                           f"Please re-extract features from cleaned_data.csv.")
        
        if remove_unknown:
            mask = ~(df[self.target_columns].isin(['Unknown']).any(axis=1))
            df = df[mask].reset_index(drop=True)
            features = features[mask]
        
        # Filter out classes with too few samples
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
        
        return X, y_dict
    
    def tune_knn(self, X_train, y_train, cv=5, n_jobs=-1):
        """Tune KNN hyperparameters"""
        print("\nTuning KNN...")
        
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            knn, param_grid, cv=cv, scoring='accuracy',
            n_jobs=n_jobs, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def tune_random_forest(self, X_train, y_train, cv=5, n_jobs=-1):
        """Tune Random Forest hyperparameters"""
        print("\nTuning Random Forest...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
        
        # Use RandomizedSearchCV for faster search
        random_search = RandomizedSearchCV(
            rf, param_grid, n_iter=20, cv=cv, scoring='accuracy',
            n_jobs=n_jobs, random_state=42, verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
    
    def tune_svm(self, X_train, y_train, cv=5, n_jobs=-1):
        """Tune SVM hyperparameters"""
        print("\nTuning SVM...")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        svm = SVC(probability=True, random_state=42)
        
        # Use RandomizedSearchCV for faster search
        random_search = RandomizedSearchCV(
            svm, param_grid, n_iter=15, cv=cv, scoring='accuracy',
            n_jobs=n_jobs, random_state=42, verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
    
    def tune_neural_network(self, X_train, y_train, cv=5):
        """Tune Neural Network hyperparameters"""
        print("\nTuning Neural Network...")
        
        param_grid = {
            'hidden_layer_sizes': [(128,), (256,), (128, 64), (256, 128), (256, 128, 64)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'batch_size': [16, 32, 64]
        }
        
        mlp = MLPClassifier(
            solver='adam', max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1
        )
        
        # Use RandomizedSearchCV for faster search
        random_search = RandomizedSearchCV(
            mlp, param_grid, n_iter=15, cv=cv, scoring='accuracy',
            random_state=42, verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
    
    def tune_all_models(self, X_train, y_train_dict):
        """Tune all models for all targets"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        all_best_models = {}
        all_tuning_results = {}
        
        for col in self.target_columns:
            print(f"\n{'='*70}")
            print(f"Tuning models for {col}")
            print('='*70)
            
            y_train = y_train_dict[col]
            best_models = {}
            tuning_results = {}
            
            # Tune KNN
            best_knn, knn_params, knn_score = self.tune_knn(X_train_scaled, y_train)
            best_models['KNN'] = best_knn
            tuning_results['KNN'] = {'params': knn_params, 'cv_score': knn_score}
            
            # Tune Random Forest
            best_rf, rf_params, rf_score = self.tune_random_forest(X_train_scaled, y_train)
            best_models['Random Forest'] = best_rf
            tuning_results['Random Forest'] = {'params': rf_params, 'cv_score': rf_score}
            
            # Tune SVM
            best_svm, svm_params, svm_score = self.tune_svm(X_train_scaled, y_train)
            best_models['SVM'] = best_svm
            tuning_results['SVM'] = {'params': svm_params, 'cv_score': svm_score}
            
            # Tune Neural Network
            best_mlp, mlp_params, mlp_score = self.tune_neural_network(X_train_scaled, y_train)
            best_models['Neural Network'] = best_mlp
            tuning_results['Neural Network'] = {'params': mlp_params, 'cv_score': mlp_score}
            
            all_best_models[col] = best_models
            all_tuning_results[col] = tuning_results
        
        self.best_models = all_best_models
        self.tuning_results = all_tuning_results
        
        return all_best_models, all_tuning_results
    
    def evaluate_tuned_models(self, X_test, y_test_dict):
        """Evaluate tuned models on test set"""
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for col in self.target_columns:
            print(f"\n{'='*70}")
            print(f"Evaluating tuned models for {col}")
            print('='*70)
            
            y_test = y_test_dict[col]
            col_results = {}
            
            for model_name, model in self.best_models[col].items():
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                col_results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': y_pred
                }
                
                print(f"\n{model_name}:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
            
            results[col] = col_results
        
        return results
    
    def compare_baseline_vs_tuned(self, baseline_results=None, tuned_results=None, output_dir='task5_outputs'):
        """Compare baseline and tuned models (optional baseline_results)"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if tuned_results is None:
            tuned_results = self.evaluate_tuned_models
        
        # Create comparison plot
        model_names = ['KNN', 'Random Forest', 'SVM', 'Neural Network']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        title = 'Tuned Models - Accuracy Comparison'
        if baseline_results:
            title = 'Baseline vs Tuned Models - Accuracy Comparison'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(self.target_columns):
            tuned_accs = []
            baseline_accs = []
            
            for model_name in model_names:
                # Get tuned accuracy
                tuned_acc = tuned_results[col][model_name]['accuracy']
                tuned_accs.append(tuned_acc)
                
                # Get baseline accuracy (if available)
                if baseline_results:
                    baseline_acc = baseline_results.get(model_name, {}).get(col, {}).get('accuracy', 0)
                    baseline_accs.append(baseline_acc)
            
            x = np.arange(len(model_names))
            if baseline_results:
                width = 0.35
                axes[idx].bar(x - width/2, baseline_accs, width, label='Baseline', 
                             color='#3498db', alpha=0.7)
                axes[idx].bar(x + width/2, tuned_accs, width, label='Tuned', 
                             color='#e74c3c', alpha=0.7)
                axes[idx].legend()
            else:
                axes[idx].bar(x, tuned_accs, color='#e74c3c', alpha=0.7)
            
            axes[idx].set_ylabel('Accuracy', fontsize=12)
            axes[idx].set_title(f'{col}', fontsize=14, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved comparison plot")
        plt.close()
    
    def save_tuning_results(self, output_dir='task5_outputs'):
        """Save hyperparameter tuning results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report_file = f'{output_dir}/hyperparameter_tuning_results.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("HYPERPARAMETER TUNING RESULTS\n")
            f.write("=" * 70 + "\n\n")
            
            for col in self.target_columns:
                f.write(f"\n{'='*70}\n")
                f.write(f"{col}\n")
                f.write('='*70 + "\n\n")
                
                for model_name in self.tuning_results[col].keys():
                    f.write(f"{model_name}:\n")
                    f.write(f"  Best Parameters: {self.tuning_results[col][model_name]['params']}\n")
                    f.write(f"  Best CV Score: {self.tuning_results[col][model_name]['cv_score']:.4f}\n")
                    f.write("\n")
        
        print(f"[OK] Saved tuning results to {report_file}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("TASK 5: HYPERPARAMETER TUNING AND OPTIMIZATION")
    print("Predicting: Weather, Time of Day, Season")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading cleaned data...")
    df = pd.read_csv('cleaned_data.csv')
    print(f"   [OK] Loaded {len(df)} samples")
    
    # Load features
    print("\n2. Loading image features...")
    features = None
    try:
        loaded_features = np.load('image_features.npy')
        print(f"   [OK] Loaded features: {loaded_features.shape}")
        
        # Check if feature size matches dataframe size
        if loaded_features.shape[0] == len(df):
            features = loaded_features
            print(f"   [OK] Feature size matches dataframe size ({len(df)} rows)")
        else:
            print(f"   [WARNING] Feature size ({loaded_features.shape[0]}) doesn't match dataframe size ({len(df)})")
            print(f"   [INFO] Re-extracting features from cleaned_data.csv...")
            extractor = ImageFeatureExtractor()
            features, _ = extractor.extract_features_from_dataframe(
                df, save_path='image_features.npy', verbose=True
            )
            print(f"   [OK] Extracted features: {features.shape}")
    except FileNotFoundError:
        print("   Features not found. Extracting from images...")
        extractor = ImageFeatureExtractor()
        features, _ = extractor.extract_features_from_dataframe(
            df, save_path='image_features.npy', verbose=True
        )
        print(f"   [OK] Extracted features: {features.shape}")
    
    if features is None:
        raise ValueError("Failed to load or extract features")
    
    # Initialize tuner
    print("\n3. Initializing hyperparameter tuner...")
    tuner = HyperparameterTuning()
    
    # Prepare data
    print("\n4. Preparing data...")
    X, y_dict = tuner.prepare_data(df, features, remove_unknown=True, min_samples_per_class=3)
    print(f"   [OK] Prepared {len(X)} samples")
    
    # Split data
    print("\n5. Splitting data...")
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    y_train_dict = {}
    y_test_dict = {}
    for col in tuner.target_columns:
        y_train_dict[col] = y_dict[col][train_idx]
        y_test_dict[col] = y_dict[col][test_idx]
    
    print(f"   [OK] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Tune models
    print("\n6. Tuning hyperparameters for all models...")
    print("   (This may take a while...)")
    best_models, tuning_results = tuner.tune_all_models(X_train, y_train_dict)
    
    # Evaluate tuned models
    print("\n7. Evaluating tuned models on test set...")
    tuned_results = tuner.evaluate_tuned_models(X_test, y_test_dict)
    
    # Save results
    print("\n8. Saving tuning results...")
    tuner.save_tuning_results()
    
    # Summary
    print("\n" + "=" * 70)
    print("TASK 5 COMPLETE!")
    print("=" * 70)
    print("\nSummary of Tuned Models:")
    print("\n" + "-" * 70)
    print(f"{'Target':<20} {'Model':<20} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 70)
    for col in tuner.target_columns:
        for model_name in tuned_results[col].keys():
            acc = tuned_results[col][model_name]['accuracy']
            f1 = tuned_results[col][model_name]['f1_score']
            print(f"{col:<20} {model_name:<20} {acc:<12.4f} {f1:<12.4f}")
    print("=" * 70)
    
    # Best model per target
    print("\nBest Model for Each Target:")
    print("-" * 70)
    for col in tuner.target_columns:
        best_model = max(tuned_results[col].keys(), 
                        key=lambda m: tuned_results[col][m]['accuracy'])
        best_acc = tuned_results[col][best_model]['accuracy']
        print(f"{col}: {best_model} (Accuracy: {best_acc:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()

