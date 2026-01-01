import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import warnings

warnings.filterwarnings('ignore')


class MultiLabelTravelPreprocessor:
    """
    Data Preprocessor for Multi-Label Image Classification
    Task: Predict weather + time of day + season + mood/emotion from images
    ENCS5341 - Assignment 3
    """

    def __init__(self):
        # Define valid categories for target variables
        # Note: "Not Clear" is a VALID class for Weather and Season
        self.VALID_WEATHER = ['Sunny', 'Rainy', 'Cloudy', 'Snowy', 'Not Clear']
        self.VALID_TIME = ['Morning', 'Afternoon', 'Evening']
        self.VALID_SEASON = ['Spring', 'Summer', 'Fall', 'Winter', 'Not Clear']
        self.VALID_MOOD = ['Excitement', 'Happiness', 'Curiosity', 'Nostalgia',
                           'Adventure', 'Romance', 'Melancholy']

        # Target labels for multi-label prediction
        self.target_columns = ['Weather', 'Time of Day', 'Season', 'Mood/Emotion']

        # Statistics tracking
        self.stats = {
            'total_rows': 0,
            'removed_rows': 0,
            'removed_reasons': [],
            'url_issues': 0,
            'missing_targets': 0,
            'not_clear_counts': {},
            'weather_fixed': 0,
            'time_fixed': 0,
            'season_fixed': 0,
            'mood_fixed': 0,
            'description_fixed': 0,
            'country_fixed': 0,
            'activity_fixed': 0
        }

        self.logs = []

    def add_log(self, message, log_type='INFO'):
        """Add a log entry"""
        log_entry = f"[{log_type}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

    def normalize_value(self, value):
        """Normalize string values"""
        if pd.isna(value) or value is None:
            return ''
        value_str = str(value).strip()
        if value_str.lower() in ['', 'nan', 'null', 'none']:
            return ''
        return value_str

    def is_valid_url_format(self, url):
        """Check if URL has valid format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def standardize_category(self, value, valid_options, default_value):
        """Standardize categorical values - CASE SENSITIVE exact matching only"""
        normalized = self.normalize_value(value)

        if not normalized:
            return default_value

        # Exact case-sensitive match
        if normalized in valid_options:
            return normalized

        # If no exact match, return default
        return default_value

    def has_valid_targets(self, row):
        """Check if row has valid target labels (not missing/unknown)"""
        missing_count = 0
        for col in self.target_columns:
            if row[col] == 'Unknown':
                missing_count += 1

        # If all 4 targets are 'Unknown', the sample is not useful
        return missing_count < 4

    def process_data(self, input_file, output_file='cleaned_travel_data.csv',
                     remove_all_unknown=True, min_class_samples=5):
        """
        Main preprocessing function for multi-label classification

        Parameters:
        -----------
        input_file : str
            Path to input CSV file
        output_file : str
            Path to save cleaned CSV file
        remove_all_unknown : bool
            Remove samples where all targets are 'Unknown' (empty/missing)
        min_class_samples : int
            Minimum number of samples per class (for filtering rare classes)
        """

        self.add_log("=" * 70)
        self.add_log("STARTING PREPROCESSING FOR MULTI-LABEL CLASSIFICATION")
        self.add_log("Task: Predict Weather + Time of Day + Season + Mood/Emotion")
        self.add_log("=" * 70)

        # Load data with encoding detection
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        df = None

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(input_file, encoding=encoding)
                self.stats['total_rows'] = len(df)
                self.add_log(f"✓ Loaded {len(df)} rows using {encoding} encoding")
                break
            except (UnicodeDecodeError, Exception):
                continue

        if df is None:
            self.add_log("✗ Failed to load file with any encoding", 'ERROR')
            return None

        self.add_log("\n" + "=" * 70)
        self.add_log("STEP 1: CLEANING AND STANDARDIZING DATA")
        self.add_log("=" * 70)

        cleaned_rows = []

        for idx, row in df.iterrows():
            removal_reason = None

            # 1. Validate Image URL
            image_url = self.normalize_value(row['Image URL'])

            if not image_url:
                removal_reason = "Empty Image URL"
                self.stats['url_issues'] += 1
            elif not self.is_valid_url_format(image_url):
                removal_reason = "Invalid URL format"
                self.stats['url_issues'] += 1

            if removal_reason:
                self.add_log(f"Row {idx + 1}: Removed - {removal_reason}", 'WARNING')
                self.stats['removed_rows'] += 1
                self.stats['removed_reasons'].append(removal_reason)
                continue

            # 2. Process Description (not a target, but useful for analysis)
            description = self.normalize_value(row['Description'])
            if not description:
                description = 'null'
                self.stats['description_fixed'] += 1

            # 3. Process Country (not a target, but useful for analysis)
            country = self.normalize_value(row['Country'])
            if not country:
                country = 'null'
                self.stats['country_fixed'] += 1

            # 4. Process TARGET: Weather
            # "Not Clear" is a VALID class for Weather (e.g., night photos, unclear conditions)
            original_weather = self.normalize_value(row['Weather'])
            weather = self.standardize_category(
                row['Weather'], self.VALID_WEATHER, 'Unknown'  # Use 'Unknown' for truly missing values
            )
            if weather != original_weather and original_weather:
                self.stats['weather_fixed'] += 1

            # 5. Process TARGET: Time of Day
            # No "Not Clear" option - empty values become "Unknown"
            original_time = self.normalize_value(row['Time of Day'])
            time_of_day = self.standardize_category(
                row['Time of Day'], self.VALID_TIME, 'Unknown'
            )
            if time_of_day != original_time and original_time:
                self.stats['time_fixed'] += 1

            # 6. Process TARGET: Season
            # "Not Clear" is a VALID class for Season (e.g., tropical locations, unclear season)
            original_season = self.normalize_value(row['Season'])
            season = self.standardize_category(
                row['Season'], self.VALID_SEASON, 'Unknown'
            )
            if season != original_season and original_season:
                self.stats['season_fixed'] += 1

            # 7. Process Activity (not a target, but keep for potential analysis)
            activity = self.normalize_value(row['Activity'])
            if not activity:
                activity = 'No Activity'
                self.stats['activity_fixed'] += 1

            # 8. Process TARGET: Mood/Emotion
            # No "Not Clear" option - empty values become "Unknown"
            original_mood = self.normalize_value(row['Mood/Emotion'])
            mood = self.standardize_category(
                row['Mood/Emotion'], self.VALID_MOOD, 'Unknown'
            )
            if mood != original_mood and original_mood:
                self.stats['mood_fixed'] += 1

            # Create cleaned row
            cleaned_row = {
                'Image URL': image_url,
                'Description': description,
                'Country': country,
                'Weather': weather,
                'Time of Day': time_of_day,
                'Season': season,
                'Activity': activity,
                'Mood/Emotion': mood
            }

            # Check if sample has valid targets (not all Unknown)
            if remove_all_unknown and not self.has_valid_targets(cleaned_row):
                self.add_log(f"Row {idx + 1}: Removed - All targets are 'Unknown'", 'WARNING')
                self.stats['removed_rows'] += 1
                self.stats['missing_targets'] += 1
                self.stats['removed_reasons'].append("All targets 'Unknown'")
                continue

            cleaned_rows.append(cleaned_row)

        # Create cleaned dataframe
        cleaned_df = pd.DataFrame(cleaned_rows)

        self.add_log("\n" + "=" * 70)
        self.add_log("STEP 2: ANALYZING TARGET LABEL DISTRIBUTIONS")
        self.add_log("=" * 70)

        # Analyze distributions
        self._analyze_distributions(cleaned_df)

        # Optional: Remove rare classes
        if min_class_samples > 1:
            self.add_log(f"\nFiltering classes with less than {min_class_samples} samples...")
            cleaned_df = self._filter_rare_classes(cleaned_df, min_class_samples)

        # Save to CSV
        cleaned_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        self.add_log(f"\n✓ Cleaned data saved to {output_file}")

        # Print final statistics
        self.print_statistics(cleaned_df)

        return cleaned_df

    def _analyze_distributions(self, df):
        """Analyze and log target label distributions"""
        for col in self.target_columns:
            counts = df[col].value_counts()
            self.add_log(f"\n{col} distribution:")
            for label, count in counts.items():
                percentage = (count / len(df)) * 100
                self.add_log(f"  {label}: {count} ({percentage:.1f}%)")
                if label in ['Not Clear', 'Unknown']:
                    self.stats['not_clear_counts'][f"{col}_{label}"] = count

    def _filter_rare_classes(self, df, min_samples):
        """Filter out samples with rare classes that have too few examples"""
        initial_count = len(df)

        for col in self.target_columns:
            value_counts = df[col].value_counts()
            # Don't remove 'Unknown' class, but remove other rare classes
            rare_classes = value_counts[value_counts < min_samples].index.tolist()

            if rare_classes and 'Unknown' not in rare_classes:
                self.add_log(f"Removing rare classes from {col}: {rare_classes}")
                df = df[~df[col].isin(rare_classes)]

        removed = initial_count - len(df)
        if removed > 0:
            self.add_log(f"Removed {removed} samples due to rare classes")
            self.stats['removed_rows'] += removed

        return df

    def print_statistics(self, cleaned_df):
        """Print comprehensive preprocessing statistics"""
        print("\n" + "=" * 70)
        print("FINAL PREPROCESSING STATISTICS")
        print("=" * 70)
        print(f"Original dataset size: {self.stats['total_rows']} rows")
        print(f"Cleaned dataset size: {len(cleaned_df)} rows")
        print(f"Total rows removed: {self.stats['removed_rows']}")
        print(f"Retention rate: {(len(cleaned_df) / self.stats['total_rows'] * 100):.2f}%")

        print("\n" + "-" * 70)
        print("REMOVAL REASONS:")
        print("-" * 70)
        if self.stats['removed_reasons']:
            from collections import Counter
            reason_counts = Counter(self.stats['removed_reasons'])
            for reason, count in reason_counts.items():
                print(f"  {reason}: {count}")

        print("\n" + "-" * 70)
        print("DATA CLEANING OPERATIONS:")
        print("-" * 70)
        print(f"URL issues: {self.stats['url_issues']}")
        print(f"Weather standardized: {self.stats['weather_fixed']}")
        print(f"Time of Day standardized: {self.stats['time_fixed']}")
        print(f"Season standardized: {self.stats['season_fixed']}")
        print(f"Mood/Emotion standardized: {self.stats['mood_fixed']}")

        print("\n" + "-" * 70)
        print("'NOT CLEAR' AND 'UNKNOWN' COUNTS:")
        print("-" * 70)
        print("Note: 'Not Clear' is a valid class for Weather and Season")
        print("      'Unknown' indicates missing/invalid data")
        print()
        for key, count in self.stats['not_clear_counts'].items():
            percentage = (count / len(cleaned_df)) * 100
            print(f"  {key}: {count} ({percentage:.1f}%)")

        print("\n" + "-" * 70)
        print("CLASS BALANCE ANALYSIS:")
        print("-" * 70)
        for col in self.target_columns:
            counts = cleaned_df[col].value_counts()
            print(f"\n{col}:")
            max_count = counts.max()
            min_count = counts.min()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"  Classes: {len(counts)}")
            print(f"  Most common: {counts.index[0]} ({counts.iloc[0]} samples)")
            print(f"  Least common: {counts.index[-1]} ({counts.iloc[-1]} samples)")
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

        print("=" * 70 + "\n")

    def generate_eda_report(self, df, output_dir='eda_outputs'):
        """Generate comprehensive EDA visualizations and report"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        self.add_log("\n" + "=" * 70)
        self.add_log("GENERATING EDA VISUALIZATIONS")
        self.add_log("=" * 70)

        # 1. Target variable distributions
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Target Variable Distributions for Multi-Label Classification',
                     fontsize=16, fontweight='bold')

        for idx, col in enumerate(self.target_columns):
            ax = axes[idx // 2, idx % 2]
            counts = df[col].value_counts()

            # Create bar plot with different colors for valid vs unknown
            colors = ['#e74c3c' if label == 'Unknown' else '#3498db' for label in counts.index]
            bars = ax.bar(range(len(counts)), counts.values, color=colors, alpha=0.7)

            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(col, fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/target_distributions.png', dpi=300, bbox_inches='tight')
        self.add_log(f"✓ Saved: {output_dir}/target_distributions.png")
        plt.close()

        # 2. Correlation heatmap (for encoded labels)
        self.add_log("\nGenerating label correlation matrix...")
        from sklearn.preprocessing import LabelEncoder

        encoded_df = df[self.target_columns].copy()
        for col in self.target_columns:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col])

        plt.figure(figsize=(10, 8))
        correlation_matrix = encoded_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Between Target Variables', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/label_correlation.png', dpi=300, bbox_inches='tight')
        self.add_log(f"✓ Saved: {output_dir}/label_correlation.png")
        plt.close()

        # 3. Country distribution
        self.add_log("\nGenerating country distribution...")
        plt.figure(figsize=(14, 8))
        country_counts = df['Country'].value_counts().head(20)
        bars = plt.barh(range(len(country_counts)), country_counts.values, color='coral', alpha=0.7)
        plt.yticks(range(len(country_counts)), country_counts.index)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Country', fontsize=12)
        plt.title('Top 20 Countries in Dataset', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height() / 2.,
                     f' {int(width)}',
                     ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/country_distribution.png', dpi=300, bbox_inches='tight')
        self.add_log(f"✓ Saved: {output_dir}/country_distribution.png")
        plt.close()

        # 4. Multi-label statistics
        self.add_log("\nCalculating multi-label statistics...")
        stats_text = []
        stats_text.append("MULTI-LABEL CLASSIFICATION DATASET STATISTICS")
        stats_text.append("=" * 70)
        stats_text.append(f"Total samples: {len(df)}")
        stats_text.append(f"Number of target variables: {len(self.target_columns)}")
        stats_text.append("")
        stats_text.append("IMPORTANT NOTES:")
        stats_text.append("- 'Not Clear' is a VALID class for Weather and Season")
        stats_text.append("- 'Unknown' represents missing/invalid data")
        stats_text.append("")

        for col in self.target_columns:
            stats_text.append(f"\n{col}:")
            stats_text.append(f"  Unique classes: {df[col].nunique()}")
            stats_text.append(f"  Classes: {', '.join(sorted(df[col].unique()))}")

            not_clear_count = (df[col] == 'Not Clear').sum()
            unknown_count = (df[col] == 'Unknown').sum()

            if not_clear_count > 0:
                not_clear_pct = not_clear_count / len(df) * 100
                stats_text.append(f"  'Not Clear' samples (valid class): {not_clear_count} ({not_clear_pct:.1f}%)")

            if unknown_count > 0:
                unknown_pct = unknown_count / len(df) * 100
                stats_text.append(f"  'Unknown' samples (missing data): {unknown_count} ({unknown_pct:.1f}%)")

        stats_file = f'{output_dir}/dataset_statistics.txt'
        with open(stats_file, 'w') as f:
            f.write('\n'.join(stats_text))
        self.add_log(f"✓ Saved: {stats_file}")

        self.add_log("\n✓ EDA report generation complete!")

    # -----------------------------
    # ADDED: Extra Preprocessing + EDA (without changing your existing logic)
    # -----------------------------
    def generate_extra_eda(self, original_df, cleaned_df, output_dir='eda_outputs'):
        """
        Extra EDA + preprocessing diagnostics required by rubric:
        - Missingness summary (before/after)
        - Duplicates (Image URL)
        - Invalid category examples
        - Unknown-rate and label-completeness analysis
        - Cross-tab "trend" heatmaps between targets
        - Quantitative diversity (entropy)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        self.add_log("\n" + "=" * 70)
        self.add_log("GENERATING EXTRA PREPROCESSING DIAGNOSTICS + EDA")
        self.add_log("=" * 70)

        # -----------------------------
        # 1) Missingness summary (before/after)
        # -----------------------------
        cols = ['Image URL', 'Description', 'Country', 'Weather', 'Time of Day', 'Season', 'Activity', 'Mood/Emotion']

        def missing_report(df):
            rep = []
            for c in cols:
                if c not in df.columns:
                    continue
                # treat empty strings as missing too
                missing = df[c].isna().sum() + (df[c].astype(str).str.strip() == '').sum()
                pct = (missing / len(df) * 100) if len(df) else 0
                rep.append((c, int(missing), pct))
            return rep

        before = missing_report(original_df)
        after = missing_report(cleaned_df)

        miss_file = os.path.join(output_dir, "missingness_before_after.txt")
        with open(miss_file, "w", encoding="utf-8") as f:
            f.write("MISSINGNESS REPORT (Before vs After)\n")
            f.write("=" * 70 + "\n\n")
            f.write("BEFORE CLEANING:\n")
            for c, m, p in before:
                f.write(f"{c}: {m} ({p:.2f}%)\n")
            f.write("\nAFTER CLEANING:\n")
            for c, m, p in after:
                f.write(f"{c}: {m} ({p:.2f}%)\n")
        self.add_log(f"✓ Saved: {miss_file}")

        # -----------------------------
        # 2) Duplicates check (Image URL)
        # -----------------------------
        dup_count = cleaned_df.duplicated(subset=['Image URL']).sum() if 'Image URL' in cleaned_df.columns else 0
        dup_file = os.path.join(output_dir, "duplicate_image_urls.txt")
        with open(dup_file, "w", encoding="utf-8") as f:
            f.write("DUPLICATE IMAGE URL CHECK\n")
            f.write("=" * 70 + "\n")
            f.write(f"Duplicate Image URL rows in cleaned_df: {int(dup_count)}\n\n")
            if dup_count > 0:
                f.write("Examples (first 20 duplicates):\n")
                dups = cleaned_df[cleaned_df.duplicated(subset=['Image URL'], keep=False)].head(20)
                f.write(dups[['Image URL', 'Weather', 'Time of Day', 'Season', 'Mood/Emotion']].to_string(index=False))
        self.add_log(f"✓ Saved: {dup_file}")

        # -----------------------------
        # 3) Invalid category examples (from original_df)
        # -----------------------------
        if all(c in original_df.columns for c in self.target_columns):
            def top_invalid(col, valid_list):
                series = original_df[col].astype(str).str.strip()
                mask_missing = series.str.lower().isin(['', 'nan', 'null', 'none'])
                series = series[~mask_missing]
                invalid = series[~series.isin(valid_list)]
                return invalid.value_counts().head(10)

            invalid_weather = top_invalid('Weather', self.VALID_WEATHER)
            invalid_time = top_invalid('Time of Day', self.VALID_TIME)
            invalid_season = top_invalid('Season', self.VALID_SEASON)
            invalid_mood = top_invalid('Mood/Emotion', self.VALID_MOOD)

            inv_file = os.path.join(output_dir, "invalid_label_examples.txt")
            with open(inv_file, "w", encoding="utf-8") as f:
                f.write("INVALID LABEL EXAMPLES (Top 10)\n")
                f.write("=" * 70 + "\n\n")

                f.write("Weather invalid examples:\n")
                f.write(invalid_weather.to_string() + "\n\n")

                f.write("Time of Day invalid examples:\n")
                f.write(invalid_time.to_string() + "\n\n")

                f.write("Season invalid examples:\n")
                f.write(invalid_season.to_string() + "\n\n")

                f.write("Mood/Emotion invalid examples:\n")
                f.write(invalid_mood.to_string() + "\n\n")
            self.add_log(f"✓ Saved: {inv_file}")

        # -----------------------------
        # 4) Unknown-rate plots (per target)
        # -----------------------------
        unknown_rates = {}
        for col in self.target_columns:
            unknown_rates[col] = (cleaned_df[col] == 'Unknown').mean() * 100

        plt.figure(figsize=(10, 6))
        plt.bar(list(unknown_rates.keys()), list(unknown_rates.values()))
        plt.xticks(rotation=30, ha='right')
        plt.ylabel("Unknown Rate (%)")
        plt.title("Unknown Rate per Target Column")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        unk_plot = os.path.join(output_dir, "unknown_rate_per_target.png")
        plt.savefig(unk_plot, dpi=300, bbox_inches='tight')
        plt.close()
        self.add_log(f"✓ Saved: {unk_plot}")

        # -----------------------------
        # 5) Label completeness per sample (#Unknown among 4 targets)
        # -----------------------------
        unknown_per_row = (cleaned_df[self.target_columns] == 'Unknown').sum(axis=1)
        counts = unknown_per_row.value_counts().sort_index()

        plt.figure(figsize=(8, 5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xlabel("# of 'Unknown' labels in sample (0 to 4)")
        plt.ylabel("Number of samples")
        plt.title("Label Completeness Distribution")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        comp_plot = os.path.join(output_dir, "label_completeness_distribution.png")
        plt.savefig(comp_plot, dpi=300, bbox_inches='tight')
        plt.close()
        self.add_log(f"✓ Saved: {comp_plot}")

        # -----------------------------
        # 6) Cross-tab “trend” heatmaps
        # -----------------------------
        def crosstab_heatmap(a, b, filename, title):
            ct = pd.crosstab(cleaned_df[a], cleaned_df[b])
            plt.figure(figsize=(10, 7))
            sns.heatmap(ct, annot=True, fmt='d', linewidths=0.5)
            plt.title(title)
            plt.tight_layout()
            out = os.path.join(output_dir, filename)
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close()
            self.add_log(f"✓ Saved: {out}")

        if 'Weather' in cleaned_df.columns and 'Time of Day' in cleaned_df.columns:
            crosstab_heatmap('Weather', 'Time of Day', "heatmap_weather_vs_time.png",
                             "Weather vs Time of Day (Counts)")

        if 'Season' in cleaned_df.columns and 'Weather' in cleaned_df.columns:
            crosstab_heatmap('Season', 'Weather', "heatmap_season_vs_weather.png",
                             "Season vs Weather (Counts)")

        if 'Mood/Emotion' in cleaned_df.columns and 'Weather' in cleaned_df.columns:
            crosstab_heatmap('Mood/Emotion', 'Weather', "heatmap_mood_vs_weather.png",
                             "Mood/Emotion vs Weather (Counts)")

        # -----------------------------
        # 7) Quantitative diversity: entropy per target
        # -----------------------------
        ent_file = os.path.join(output_dir, "target_entropy.txt")
        with open(ent_file, "w", encoding="utf-8") as f:
            f.write("TARGET DIVERSITY (Shannon Entropy)\n")
            f.write("=" * 70 + "\n\n")
            for col in self.target_columns:
                p = cleaned_df[col].value_counts(normalize=True)
                entropy = -(p * np.log2(p + 1e-12)).sum()
                f.write(f"{col}: {entropy:.4f}\n")
        self.add_log(f"✓ Saved: {ent_file}")

        self.add_log("\n✓ Extra EDA + diagnostics complete!")

    def save_logs(self, log_file='preprocessing_logs.txt'):
        """Save logs to a text file"""
        with open(log_file, 'w', encoding='utf-8') as f:
            for log in self.logs:
                f.write(log + '\n')
        print(f"\n✓ Logs saved to {log_file}")


def main():
    """Main execution function"""

    # Initialize preprocessor
    preprocessor = MultiLabelTravelPreprocessor()

    # File paths
    input_file = 'C:/Users/Asus/Downloads/filtered_Data_final_final_layan.csv'  # Update with your path
    output_file = 'cleaned_travel_data_98989898.csv'

    print("\n" + "=" * 70)
    print("MULTI-LABEL IMAGE CLASSIFICATION PREPROCESSING")
    print("Task: Predict Weather + Time of Day + Season + Mood/Emotion from Images")
    print("=" * 70)
    print("\nIMPORTANT:")
    print("- 'Not Clear' is a VALID class for Weather and Season")
    print("- 'Unknown' is used for missing/invalid data")
    print("=" * 70 + "\n")

    # Process the data
    cleaned_df = preprocessor.process_data(
        input_file=input_file,
        output_file=output_file,
        remove_all_unknown=True,  # Remove samples where all targets are 'Unknown'
        min_class_samples=5  # Minimum samples per class (set to 1 to keep all)
    )

    if cleaned_df is not None:
        # Generate EDA report
        preprocessor.generate_eda_report(cleaned_df, output_dir='eda_outputs')

        # ADDED: Extra EDA + diagnostics (rubric: trends + issues + quantitative summaries)
        # This reads the original file again only for BEFORE/AFTER comparisons.
        # If utf-8 fails, change the encoding to the one that worked for you earlier (latin-1/cp1252/etc).
        try:
            original_df_for_report = pd.read_csv(input_file, encoding='utf-8')
        except Exception:
            original_df_for_report = pd.read_csv(input_file, encoding='latin-1')

        preprocessor.generate_extra_eda(original_df_for_report, cleaned_df, output_dir='eda_outputs')

        # Save logs
        preprocessor.save_logs('preprocessing_logs.txt')

        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE!")
        print("=" * 70)
        print(f"✓ Cleaned dataset: {output_file}")
        print(f"✓ Dataset shape: {cleaned_df.shape}")
        print(f"✓ EDA visualizations: eda_outputs/")
        print(f"✓ Logs: preprocessing_logs.txt")
        print("\n" + "=" * 70)
        print("READY FOR MODEL TRAINING!")
        print("=" * 70)

        # Display sample
        print("\nFirst 3 rows of cleaned data:")
        print(cleaned_df.head(3))

        # Show class distributions
        print("\n" + "=" * 70)
        print("FINAL CLASS DISTRIBUTIONS:")
        print("=" * 70)
        for col in preprocessor.target_columns:
            print(f"\n{col}:")
            print(cleaned_df[col].value_counts())


if __name__ == "__main__":
    main()
