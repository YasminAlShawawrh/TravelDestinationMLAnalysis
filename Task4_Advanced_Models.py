"""
Task 4: Advanced Machine Learning Models
Predict Weather, Time of Day, and Season using:
- Random Forest
- Support Vector Machine (SVM)
- CNN (Convolutional Neural Network) - processes images directly
ENCS5341 - Assignment 3
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
        # Load image from URL
        url = self.urls[idx]
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, timeout=5, headers=headers, verify=False, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except:
            # Return black image if loading fails
            img = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            img = self.transform(img)
        
        # Return image and labels for all targets
        labels = {key: val[idx] for key, val in self.labels_dict.items()}
        return img, labels


class CNNModel(nn.Module):
    """CNN model for multi-target classification"""
    
    def __init__(self, num_classes_dict):
        """
        Initialize CNN model
        
        Parameters:
        -----------
        num_classes_dict : dict
            Dictionary mapping target names to number of classes
        """
        super(CNNModel, self).__init__()
        
        # Base CNN layers (using ResNet-like architecture)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Separate heads for each target
        self.heads = nn.ModuleDict()
        for target_name, num_classes in num_classes_dict.items():
            self.heads[target_name] = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """Create a residual layer"""
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
        # Base CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Separate predictions for each target
        outputs = {}
        for target_name in self.heads.keys():
            outputs[target_name] = self.heads[target_name](x)
        
        return outputs


class AdvancedMLModels:
    """Advanced ML models for multi-label classification"""
    
    def __init__(self):
        """Initialize advanced models"""
        self.models = {
            'Random Forest': {},
            'SVM': {},
            'CNN': None  # CNN is a single model for all targets
        }
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_columns = ['Weather', 'Time of Day', 'Season']
        
    def prepare_data(self, df, features, remove_unknown=True, min_samples_per_class=3):
        """
        Prepare data for training
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with target labels
        features : np.ndarray
            Image features
        remove_unknown : bool
            Remove samples with 'Unknown' labels
        min_samples_per_class : int
            Minimum samples per class to keep the class
            
        Returns:
        --------
        X, y_dict : tuple
            Features and target labels dictionary
        """
        # Ensure features and dataframe have the same number of rows
        if len(features) != len(df):
            raise ValueError(f"Feature array size ({len(features)}) doesn't match dataframe size ({len(df)}). "
                           f"Please re-extract features from cleaned_data.csv.")
        
        # Filter out rows with 'Unknown' in any target if requested
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
        
        # Encode labels
        y_dict = {}
        for col in self.target_columns:
            le = LabelEncoder()
            y_dict[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        return X, y_dict
    
    def train_models(self, X_train, y_train_dict):
        """
        Train all advanced models
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train_dict : dict
            Dictionary of training labels for each target
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train models for each target
        for col in self.target_columns:
            print(f"\n{'='*70}")
            print(f"Training models for {col}...")
            print('='*70)
            
            # Random Forest
            print(f"\n1. Training Random Forest for {col}...")
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train_dict[col])
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train_dict[col])
            class_weight_dict = dict(zip(classes, class_weights))
            
            rf = RandomForestClassifier(
                n_estimators=150,  # Increased for better performance
                max_depth=25,  # Increased depth
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight='balanced',  # Handle class imbalance
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train_dict[col])
            self.models['Random Forest'][col] = rf
            print(f"   [OK] Random Forest trained")
            
            # SVM
            print(f"\n2. Training SVM for {col}...")
            svm = SVC(
                kernel='rbf',
                C=10.0,  # Increased C for better performance
                gamma='scale',
                class_weight='balanced',  # Handle class imbalance
                probability=True,
                random_state=42
            )
            svm.fit(X_train_scaled, y_train_dict[col])
            self.models['SVM'][col] = svm
            print(f"   [OK] SVM trained")
            
            # Note: CNN will be trained separately after all other models
    
    def train_cnn(self, df, urls, y_train_dict, y_test_dict, train_indices, test_indices, 
                  epochs=10, batch_size=32, learning_rate=0.001):
        """
        Train CNN model on images directly
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataframe
        urls : list
            List of image URLs
        y_train_dict : dict
            Training labels dictionary
        y_test_dict : dict
            Test labels dictionary
        train_indices : np.ndarray
            Training indices
        test_indices : np.ndarray
            Test indices
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        """
        print(f"\n{'='*70}")
        print("Training CNN (Convolutional Neural Network) on images...")
        print("=" * 70)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Get number of classes for each target
        num_classes_dict = {}
        for col in self.target_columns:
            num_classes_dict[col] = len(self.label_encoders[col].classes_)
        
        # Create CNN model
        model = CNNModel(num_classes_dict).to(device)
        
        # Create datasets with proper label indexing
        train_urls = [urls[i] for i in train_indices]
        test_urls = [urls[i] for i in test_indices]
        
        # Create label dictionaries with proper indexing
        train_labels = {}
        test_labels = {}
        for col in self.target_columns:
            train_labels[col] = [y_train_dict[col][idx] for idx in range(len(train_indices))]
            test_labels[col] = [y_test_dict[col][idx] for idx in range(len(test_indices))]
        
        train_dataset = ImageDataset(train_urls, train_labels)
        test_dataset = ImageDataset(test_urls, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        print(f"\nTraining for {epochs} epochs...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss for each target
                loss = 0
                for col in self.target_columns:
                    # Labels come as a list from the dataset
                    target_labels = torch.tensor(labels[col], dtype=torch.long).to(device)
                    loss += criterion(outputs[col], target_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            avg_loss = running_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.4f}")
        
        # Save model
        self.models['CNN'] = model
        self.cnn_device = device
        print(f"\n[OK] CNN trained successfully")
    
    def predict(self, X, model_name, urls=None, indices=None):
        """
        Predict labels using a specific model
        
        Parameters:
        -----------
        X : np.ndarray
            Features (not used for CNN)
        model_name : str
            Name of model to use
        urls : list, optional
            Image URLs (required for CNN)
        indices : np.ndarray, optional
            Indices (required for CNN)
            
        Returns:
        --------
        dict
            Predictions for each target
        """
        if model_name == 'CNN':
            # CNN prediction
            if urls is None or indices is None:
                raise ValueError("URLs and indices required for CNN prediction")
            
            predictions = {}
            model = self.models['CNN']
            model.eval()
            
            # Create dataset and dataloader
            labels_dict = {col: np.zeros(len(indices)) for col in self.target_columns}
            dataset = ImageDataset([urls[i] for i in indices], labels_dict)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
            
            all_preds = {col: [] for col in self.target_columns}
            
            with torch.no_grad():
                for images, _ in dataloader:
                    images = images.to(self.cnn_device)
                    outputs = model(images)
                    
                    for col in self.target_columns:
                        preds = torch.argmax(outputs[col], dim=1).cpu().numpy()
                        all_preds[col].extend(preds)
            
            # Convert to labels
            for col in self.target_columns:
                pred_encoded = np.array(all_preds[col])
                pred_labels = self.label_encoders[col].inverse_transform(pred_encoded)
                predictions[col] = pred_labels
        else:
            # Traditional models (Random Forest, SVM)
            X_scaled = self.scaler.transform(X)
            predictions = {}
            
            for col in self.target_columns:
                model = self.models[model_name][col]
                pred_encoded = model.predict(X_scaled)
                pred_labels = self.label_encoders[col].inverse_transform(pred_encoded)
                predictions[col] = pred_labels
        
        return predictions
    
    def evaluate(self, X_test, y_test_dict, test_urls=None, test_indices=None):
        """
        Evaluate all models
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features (not used for CNN)
        y_test_dict : dict
            Test labels dictionary
        test_urls : list, optional
            Test image URLs (required for CNN)
        test_indices : np.ndarray, optional
            Test indices (required for CNN)
            
        Returns:
        --------
        dict
            Evaluation results for all models
        """
        all_results = {}
        
        for model_name in self.models.keys():
            if model_name == 'CNN' and self.models['CNN'] is None:
                continue  # Skip CNN if not trained
            
            print(f"\n{'='*70}")
            print(f"Evaluating {model_name}...")
            print('='*70)
            
            if model_name == 'CNN':
                predictions = self.predict(X_test, model_name, urls=test_urls, indices=test_indices)
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
        """Plot comparison of all models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Accuracy comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Model Comparison - Accuracy Scores', fontsize=16, fontweight='bold')
        
        # Filter out untrained models
        model_names = [name for name in self.models.keys() 
                      if name != 'CNN' or self.models['CNN'] is not None]
        model_names = [name for name in model_names if name in all_results]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, col in enumerate(self.target_columns):
            accuracies = [all_results[model][col]['accuracy'] for model in model_names]
            
            bars = axes[idx].bar(model_names, accuracies, color=colors, alpha=0.7)
            axes[idx].set_ylabel('Accuracy', fontsize=12)
            axes[idx].set_title(f'{col}', fontsize=14, fontweight='bold')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width() / 2., height,
                              f'{height:.3f}',
                              ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved comparison plot to {output_dir}/model_comparison.png")
        plt.close()
        
        # Confusion matrices for best model per target
        for col in self.target_columns:
            # Find best model for this target
            best_model = max(model_names, 
                           key=lambda m: all_results[m][col]['accuracy'])
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Confusion Matrices - {col}', fontsize=16, fontweight='bold')
            
            for idx, model_name in enumerate(model_names):
                y_true = all_results[model_name][col]['true_labels']
                y_pred = all_results[model_name][col]['predictions']
                y_true_encoded = self.label_encoders[col].transform(y_true)
                y_pred_encoded = self.label_encoders[col].transform(y_pred)
                
                cm = confusion_matrix(y_true_encoded, y_pred_encoded)
                labels = self.label_encoders[col].classes_
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels,
                           ax=axes[idx], cbar_kws={'label': 'Count'})
                axes[idx].set_title(f'{model_name}\nAccuracy: {all_results[model_name][col]["accuracy"]:.4f}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
                plt.setp(axes[idx].get_xticklabels(), rotation=45, ha='right')
                plt.setp(axes[idx].get_yticklabels(), rotation=0)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/confusion_matrices_{col.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"[OK] Saved confusion matrices for {col}")
            plt.close()
    
    def generate_reports(self, all_results, X_test, y_test_dict, output_dir='task4_outputs'):
        """Generate detailed classification reports"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report_file = f'{output_dir}/detailed_reports.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("DETAILED CLASSIFICATION REPORTS - ADVANCED MODELS\n")
            f.write("=" * 70 + "\n\n")
            
            for model_name in self.models.keys():
                if model_name == 'CNN' and self.models['CNN'] is None:
                    continue
                if model_name not in all_results:
                    continue
                    
                f.write(f"\n{'='*70}\n")
                f.write(f"{model_name.upper()}\n")
                f.write('='*70 + "\n\n")
                
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
                    f.write(report)
                    f.write("\n")
        
        print(f"[OK] Saved detailed reports to {report_file}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("TASK 4: ADVANCED MACHINE LEARNING MODELS")
    print("Predicting: Weather, Time of Day, Season")
    print("Models: Random Forest, SVM, Neural Network")
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
    
    # Initialize models
    print("\n3. Initializing advanced models...")
    models = AdvancedMLModels()
    
    # Prepare data
    print("\n4. Preparing data...")
    X, y_dict = models.prepare_data(df, features, remove_unknown=True, min_samples_per_class=3)
    print(f"   [OK] Prepared {len(X)} samples")
    
    # Get URLs aligned with filtered data
    # We need to apply the same filtering to URLs as was done in prepare_data
    filtered_df = df.copy()
    if True:  # remove_unknown was True
        mask = ~(filtered_df[models.target_columns].isin(['Unknown']).any(axis=1))
        filtered_df = filtered_df[mask].reset_index(drop=True)
    
    for col in models.target_columns:
        value_counts = filtered_df[col].value_counts()
        rare_classes = value_counts[value_counts < 3].index.tolist()
        if rare_classes:
            mask = ~filtered_df[col].isin(rare_classes)
            filtered_df = filtered_df[mask].reset_index(drop=True)
    
    urls = filtered_df['Image URL'].tolist()
    print(f"   [OK] URLs aligned with filtered data: {len(urls)} URLs")
    
    # Split data
    print("\n5. Splitting data...")
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    y_train_dict = {}
    y_test_dict = {}
    for col in models.target_columns:
        y_train_dict[col] = y_dict[col][train_idx]
        y_test_dict[col] = y_dict[col][test_idx]
    
    print(f"   [OK] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train traditional models (Random Forest, SVM)
    print("\n6. Training traditional models (Random Forest, SVM)...")
    models.train_models(X_train, y_train_dict)
    
    # Train CNN on images
    print("\n7. Training CNN on images...")
    models.train_cnn(
        df=df,
        urls=urls,
        y_train_dict=y_train_dict,
        y_test_dict=y_test_dict,
        train_indices=train_idx,
        test_indices=test_idx,
        epochs=15,  # Number of epochs
        batch_size=16,  # Smaller batch size for memory efficiency
        learning_rate=0.001
    )
    
    # Evaluate all models
    print("\n8. Evaluating all models...")
    test_urls = [urls[i] for i in test_idx]
    all_results = models.evaluate(X_test, y_test_dict, test_urls=test_urls, test_indices=test_idx)
    
    # Generate reports
    print("\n9. Generating detailed reports...")
    models.generate_reports(all_results, X_test, y_test_dict)
    
    # Plot comparisons
    print("\n10. Generating visualizations...")
    models.plot_comparison(all_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("TASK 4 COMPLETE!")
    print("=" * 70)
    print("\nSummary of Results:")
    print("\n" + "-" * 70)
    print(f"{'Model':<20} {'Weather':<12} {'Time of Day':<15} {'Season':<12}")
    print("-" * 70)
    for model_name in models.models.keys():
        if model_name == 'CNN' and models.models['CNN'] is None:
            continue
        if model_name in all_results:
            accs = [all_results[model_name][col]['accuracy'] for col in models.target_columns]
            print(f"{model_name:<20} {accs[0]:<12.4f} {accs[1]:<15.4f} {accs[2]:<12.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

