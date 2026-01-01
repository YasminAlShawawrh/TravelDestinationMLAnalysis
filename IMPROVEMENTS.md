# Improvements Made to Tasks 3, 4, and 5

## Fixed Issues

### 1. Task 3 Error Fix
- **Problem**: `ValueError: y should be a 1d array` in `plot_confusion_matrices`
- **Solution**: Fixed the function to properly extract predictions from the results dictionary and ensure it's a numpy array

### 2. Image Feature Extraction Improvements
- **Better URL handling**: Added support for more URL types and better error handling
- **User-Agent headers**: Added headers to avoid 403 Forbidden errors
- **SSL handling**: Added SSL verification bypass for problematic certificates
- **Failed image handling**: Instead of using zero vectors for failed images, now uses mean of successful features for better data quality
- **URL filtering**: Skips non-image URLs (Google Drive, YouTube, etc.)

### 3. Preprocessing Improvements
- **Fuzzy matching**: Added case-insensitive matching and common variations for categorical values
- **Better filtering**: Improved filtering to require at least 2 out of 3 prediction targets (Weather, Time of Day, Season) to be valid
- **Rare class filtering**: Added minimum samples per class filtering in all tasks

## Model Improvements

### Task 3 (Baseline KNN)
- Increased `n_neighbors` from 5 to 7
- Added `weights='distance'` for better performance on imbalanced data
- Better data filtering with minimum samples per class

### Task 4 (Advanced Models)
- **Random Forest**:
  - Increased `n_estimators` from 100 to 150
  - Increased `max_depth` from 20 to 25
  - Added `class_weight='balanced'` for handling class imbalance
  
- **SVM**:
  - Increased `C` from 1.0 to 10.0
  - Added `class_weight='balanced'`
  
- **Neural Network**:
  - Increased network depth: (512, 256, 128) instead of (256, 128)
  - Reduced regularization (`alpha=0.0001`)
  - Increased `max_iter` from 500 to 1000
  - Larger batch size (64 instead of 32)

### Task 5 (Hyperparameter Tuning)
- Updated to use improved data preparation with rare class filtering

## Expected Improvements

1. **Better accuracy**: Class weights and improved models should handle imbalanced data better
2. **More robust feature extraction**: Better handling of failed images
3. **Better data quality**: Improved preprocessing and filtering
4. **Fewer errors**: Fixed the plotting error in Task 3

## Running the Improved Code

The improvements are automatically applied. Simply run:
```bash
python Task3_Baseline_KNN.py
python Task4_Advanced_Models.py
python Task5_Hyperparameter_Tuning.py
```

Or use the runner:
```bash
python run_all_tasks.py
```

Note: You may want to re-run preprocessing if you want to benefit from the improved fuzzy matching:
```bash
python preProccessing_Task2.py
```

