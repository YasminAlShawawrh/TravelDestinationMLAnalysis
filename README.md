# Travel Destination ML Analysis

A full machine learning pipeline applied to a travel destination image dataset. The project covers data preprocessing, exploratory data analysis, baseline KNN classification, advanced models (Random Forest, SVM, CNN), and deep error analysis. 

---

## Table of contents

- [Project overview](#project-overview)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Task breakdown](#task-breakdown)
- [Output files](#output-files)
- [Technologies used](#technologies-used)
- [How to run](#how-to-run)

---

## Project overview

Given a dataset of travel destination images with metadata, the goal is to predict multiple labels simultaneously — **Weather**, **Time of Day**, and **Season** — from image features. The project progresses from preprocessing through to advanced model comparison and error analysis.

---

## Dataset

- **File:** `data.csv`
- **Columns:** `Image URL`, `Description`, `Country`, `Weather`, `Time of Day`, `Season`, `Activity`, `Mood/Emotion`
- **Targets:** `Weather` · `Time of Day` · `Season`
- **Image features** are extracted from URLs and saved as `image_features.npy`

Valid label categories:

| Column | Valid values |
|---|---|
| Weather | Sunny, Rainy, Cloudy, Snowy, Not Clear |
| Time of Day | Morning, Afternoon, Evening |
| Season | Spring, Summer, Fall, Winter, Not Clear |
| Mood/Emotion | Excitement, Happiness, Curiosity, Nostalgia, Adventure, Romance, Melancholy |

---

## Project structure

```
Travel_Destination_ML_Analysis/
├── data.csv                        # Raw input dataset
├── cleaned_data.csv                # Output of Task 2 preprocessing
├── image_features.npy              # Extracted image feature vectors
├── image_feature_extractor.py      # Feature extraction module
├── task2_preprocessing.py          # Data cleaning & EDA
├── task3_baseline_knn.py           # Baseline KNN model
├── task4_advanced_models.py        # Random Forest, SVM, CNN
└── task5_performance_analysis.py  # Error analysis on best model                
```

---

## Task breakdown

### Task 2 — Data preprocessing & EDA

Cleans the raw dataset through a multi-step validation pipeline:

- Validates image URLs (format check + optional HTTP accessibility check)
- Fuzzy-matches and normalizes Weather, Time of Day, Season, and Mood/Emotion values against valid label sets
- Fills missing Description, Country, and Activity fields with default values
- Reports retention rate, removal reasons, and per-field validation statistics
- Generates EDA visualizations: histograms, pie charts, co-occurrence heatmaps, and a label correlation matrix

**Output:** `cleaned_data.csv` + plots in `eda_outputs/`

---

### Task 3 — Baseline KNN model

Trains a K-Nearest Neighbors classifier using extracted image features.

- Evaluates k=1 and k=3 with Euclidean distance (Minkowski p=2)
- Applies `StandardScaler` before fitting
- Runs 5-fold cross-validation per target per k value
- Generates confusion matrices and a side-by-side k=1 vs k=3 accuracy/F1 comparison plot

**Targets:** Weather · Time of Day · Season  
**Output:** confusion matrices + comparison bar chart in `task3_outputs/`

---

### Task 4 — Advanced models

Trains and compares three models on the same image features:

| Model | Details |
|---|---|
| Random Forest | 150 trees, max depth 25, balanced class weights |
| SVM | RBF kernel, C=10, `class_weight='balanced'` |
| CNN | Custom architecture: 3 conv blocks → adaptive pooling → per-target classification heads |

The CNN trains directly on raw images fetched from URLs using PyTorch, with separate output heads for each target. Models are compared by accuracy and F1-score across all three targets.

**Output:** accuracy comparison chart + per-target confusion matrices + detailed classification reports in `task4_outputs/`

---

### Task 5 — Performance analysis

Deep error analysis on the best-performing model (Random Forest).

For each target:
- Generates confusion matrix and full classification report
- Identifies the lowest-recall classes (hardest to predict)
- Lists the most confused class pairs (e.g. Sunny → Cloudy)
- Exports a CSV of all misclassified examples with image URLs and descriptions

**Output:** per-class recall tables, confused-pair lists, misclassified CSVs, and summary table in `task5_outputs/`

---

## Output files

```
eda_outputs/
├── target_distributions.png
├── pie_charts.png
├── cooccurrence_heatmaps.png
├── label_correlation.png
├── country_distribution.png
├── activity_distribution.png
└── quantitative_summary.txt

task3_outputs/
├── confusion_matrices_knn_k1.png
├── confusion_matrices_knn_k3.png
└── knn_comparison_k1_vs_k3.png

task4_outputs/
├── model_comparison.png
├── confusion_matrices_Weather.png
├── confusion_matrices_Time_of_Day.png
├── confusion_matrices_Season.png
└── detailed_reports.txt

task5_outputs/
├── confusion_matrix_RF_*.png
├── classification_report_RF_*.txt
├── per_class_recall_RF_*.csv
├── most_confused_pairs_RF_*.txt
├── misclassified_examples_RF_*.csv
└── task5_summary_RF.csv
```

---

## Technologies used

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading, processing, feature handling |
| `scikit-learn` | KNN, Random Forest, SVM, scaling, metrics |
| `torch` / `torchvision` | CNN model, image transforms, DataLoader |
| `matplotlib` / `seaborn` | All visualizations and heatmaps |
| `PIL` / `requests` | Image downloading and validation |
| `urllib.parse` | URL format validation |

---

## How to run

1. Clone the repository:
```bash
git clone https://github.com/YasminAlShawawrh/Travel_Destination_ML_Analysis.git
cd Travel_Destination_ML_Analysis
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch torchvision pillow requests
```

3. Run each task in order:
```bash
python task2_preprocessing.py        # Clean data + EDA
python task3_baseline_knn.py         # Baseline KNN
python task4_advanced_models.py      # RF, SVM, CNN
python task5_performance_analysis.py # Error analysis
```

> Tasks 3–5 require `cleaned_data.csv` and `image_features.npy` generated by Task 2.  
> CNN training in Task 4 requires an internet connection to fetch images from URLs.
