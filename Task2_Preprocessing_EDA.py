"""
Task 2: Data Preprocessing and Exploratory Data Analysis (EDA)
ENCS5341 - Assignment 3

Requirements:
- Clean, preprocess, and organize dataset
- Validate all columns against valid options
- Validate image URLs work
- Perform comprehensive EDA with quantitative summaries and visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import requests
from PIL import Image
from io import BytesIO
import warnings
import os
from collections import Counter
warnings.filterwarnings('ignore')


class Task2Preprocessor:
    """Preprocessor for Task 2 with strict validation and comprehensive EDA"""
    
    def __init__(self):
        # Define valid options for each column
        self.VALID_WEATHER = ['Sunny', 'Rainy', 'Cloudy', 'Snowy', 'Not Clear']
        self.VALID_TIME = ['Morning', 'Afternoon', 'Evening']
        self.VALID_SEASON = ['Spring', 'Summer', 'Fall', 'Winter', 'Not Clear']
        self.VALID_MOOD = ['Excitement', 'Happiness', 'Curiosity', 'Nostalgia', 
                          'Adventure', 'Romance', 'Melancholy']
        
        # Statistics tracking
        self.stats = {
            'original_rows': 0,
            'final_rows': 0,
            'removed_rows': 0,
            'removal_reasons': Counter(),
            'url_validation_failed': 0,
            'invalid_weather': 0,
            'invalid_time': 0,
            'invalid_season': 0,
            'invalid_mood': 0,
            'valid_urls': 0,
            'invalid_urls': 0
        }
        
        self.logs = []
    
    def log(self, message, level='INFO'):
        """Add log entry"""
        log_entry = f"[{level}] {message}"
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
    
    def validate_image_url(self, url, timeout=5):
        """
        Validate that image URL is accessible and returns an image
        
        Parameters:
        -----------
        url : str
            Image URL
        timeout : int
            Request timeout in seconds
            
        Returns:
        --------
        bool
            True if URL is valid and accessible
        """
        if not url or not self.is_valid_url_format(url):
            return False
        
        # Skip non-image URLs
        if any(skip in url.lower() for skip in ['drive.google.com', 'youtube.com', 'vimeo.com', '.html', '.pdf']):
            return False
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=timeout, headers=headers, stream=True, verify=False)
            
            if response.status_code == 200:
                # Try to open as image
                try:
                    img = Image.open(BytesIO(response.content))
                    img.verify()  # Verify it's a valid image
                    return True
                except:
                    return False
            return False
        except Exception:
            return False
    
    def validate_weather(self, value):
        """Validate Weather column with fuzzy matching"""
        normalized = self.normalize_value(value)
        if not normalized:
            return None, 'Empty'
        
        # Exact match
        if normalized in self.VALID_WEATHER:
            return normalized, None
        
        # Handle variations
        normalized_lower = normalized.lower()
        variations = {
            'clear': 'Sunny',
            'sun': 'Sunny',
            'sunny': 'Sunny',
            'rain': 'Rainy',
            'rainy': 'Rainy',
            'cloud': 'Cloudy',
            'cloudy': 'Cloudy',
            'overcast': 'Cloudy',
            'snow': 'Snowy',
            'snowy': 'Snowy',
            'not clear': 'Not Clear',
            'unclear': 'Not Clear',
            'night': 'Not Clear'
        }
        
        # Check for variations
        for key, valid_value in variations.items():
            if key in normalized_lower:
                return valid_value, None
        
        return None, f'Invalid: {normalized}'
    
    def validate_time(self, value):
        """Validate Time of Day column with fuzzy matching"""
        normalized = self.normalize_value(value)
        if not normalized:
            return None, 'Empty'
        
        # Exact match
        if normalized in self.VALID_TIME:
            return normalized, None
        
        # Handle variations
        normalized_lower = normalized.lower()
        variations = {
            'morning': 'Morning',
            'am': 'Morning',
            'dawn': 'Morning',
            'afternoon': 'Afternoon',
            'pm': 'Afternoon',
            'midday': 'Afternoon',
            'evening': 'Evening',
            'night': 'Evening',
            'dusk': 'Evening',
            'sunset': 'Evening'
        }
        
        # Check for variations
        for key, valid_value in variations.items():
            if key in normalized_lower:
                return valid_value, None
        
        return None, f'Invalid: {normalized}'
    
    def validate_season(self, value):
        """Validate Season column with fuzzy matching"""
        normalized = self.normalize_value(value)
        if not normalized:
            return None, 'Empty'
        
        # Exact match
        if normalized in self.VALID_SEASON:
            return normalized, None
        
        # Handle variations
        normalized_lower = normalized.lower()
        variations = {
            'spring': 'Spring',
            'springtime': 'Spring',
            'summer': 'Summer',
            'summertime': 'Summer',
            'fall': 'Fall',
            'autumn': 'Fall',
            'winter': 'Winter',
            'wintertime': 'Winter',
            'not clear': 'Not Clear',
            'unclear': 'Not Clear'
        }
        
        # Check for variations
        for key, valid_value in variations.items():
            if key in normalized_lower:
                return valid_value, None
        
        return None, f'Invalid: {normalized}'
    
    def validate_mood(self, value):
        """Validate Mood/Emotion column with fuzzy matching"""
        normalized = self.normalize_value(value)
        if not normalized:
            return None, 'Empty'
        
        # Exact match
        if normalized in self.VALID_MOOD:
            return normalized, None
        
        # Handle variations (case-insensitive)
        normalized_lower = normalized.lower()
        variations = {
            'excitement': 'Excitement',
            'excited': 'Excitement',
            'happiness': 'Happiness',
            'happy': 'Happiness',
            'curiosity': 'Curiosity',
            'curious': 'Curiosity',
            'nostalgia': 'Nostalgia',
            'nostalgic': 'Nostalgia',
            'adventure': 'Adventure',
            'adventurous': 'Adventure',
            'romance': 'Romance',
            'romantic': 'Romance',
            'melancholy': 'Melancholy',
            'sad': 'Melancholy',
            'awe': 'Excitement'  # Map "Awe" to closest valid option
        }
        
        # Check for variations
        for key, valid_value in variations.items():
            if key == normalized_lower:
                return valid_value, None
        
        return None, f'Invalid: {normalized}'
    
    def preprocess_data(self, input_file='data.csv', output_file='cleaned_data.csv', 
                       validate_urls=True, verbose=True):
        """
        Preprocess data with strict validation
        
        Parameters:
        -----------
        input_file : str
            Input CSV file
        output_file : str
            Output CSV file
        validate_urls : bool
            Whether to validate image URLs
        verbose : bool
            Print progress
        """
        self.log("=" * 70)
        self.log("TASK 2: DATA PREPROCESSING AND EDA")
        self.log("=" * 70)
        
        # Load data
        self.log(f"\nLoading data from {input_file}...")
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(input_file, encoding='latin-1')
            except:
                df = pd.read_csv(input_file, encoding='cp1252')
        
        self.stats['original_rows'] = len(df)
        self.log(f"Loaded {len(df)} rows")
        
        # Process each row
        self.log("\nProcessing and validating data...")
        cleaned_rows = []
        
        for idx, row in df.iterrows():
            removal_reason = None
            
            # 1. Validate Image URL
            image_url = self.normalize_value(row['Image URL'])
            if not image_url:
                removal_reason = "Empty Image URL"
                self.stats['removal_reasons']['Empty Image URL'] += 1
            elif not self.is_valid_url_format(image_url):
                removal_reason = "Invalid URL format"
                self.stats['removal_reasons']['Invalid URL format'] += 1
            elif validate_urls:
                if not self.validate_image_url(image_url):
                    removal_reason = "Image URL not accessible"
                    self.stats['url_validation_failed'] += 1
                    self.stats['removal_reasons']['Image URL not accessible'] += 1
                else:
                    self.stats['valid_urls'] += 1
            else:
                self.stats['valid_urls'] += 1
            
            if removal_reason:
                self.stats['removed_rows'] += 1
                if verbose and idx < 10:  # Only log first 10 for brevity
                    self.log(f"Row {idx+1}: Removed - {removal_reason}", 'WARNING')
                continue
            
            # 2. Validate Weather
            weather, weather_error = self.validate_weather(row['Weather'])
            if weather_error:
                removal_reason = f"Weather: {weather_error}"
                self.stats['invalid_weather'] += 1
                self.stats['removal_reasons'][removal_reason] += 1
                self.stats['removed_rows'] += 1
                if verbose and idx < 10:
                    self.log(f"Row {idx+1}: Removed - {removal_reason}", 'WARNING')
                continue
            
            # 3. Validate Time of Day
            time_of_day, time_error = self.validate_time(row['Time of Day'])
            if time_error:
                removal_reason = f"Time of Day: {time_error}"
                self.stats['invalid_time'] += 1
                self.stats['removal_reasons'][removal_reason] += 1
                self.stats['removed_rows'] += 1
                if verbose and idx < 10:
                    self.log(f"Row {idx+1}: Removed - {removal_reason}", 'WARNING')
                continue
            
            # 4. Validate Season
            season, season_error = self.validate_season(row['Season'])
            if season_error:
                removal_reason = f"Season: {season_error}"
                self.stats['invalid_season'] += 1
                self.stats['removal_reasons'][removal_reason] += 1
                self.stats['removed_rows'] += 1
                if verbose and idx < 10:
                    self.log(f"Row {idx+1}: Removed - {removal_reason}", 'WARNING')
                continue
            
            # 5. Validate Mood/Emotion
            mood, mood_error = self.validate_mood(row['Mood/Emotion'])
            if mood_error:
                removal_reason = f"Mood/Emotion: {mood_error}"
                self.stats['invalid_mood'] += 1
                self.stats['removal_reasons'][removal_reason] += 1
                self.stats['removed_rows'] += 1
                if verbose and idx < 10:
                    self.log(f"Row {idx+1}: Removed - {removal_reason}", 'WARNING')
                continue
            
            # 6. Process other columns (Description, Country, Activity)
            description = self.normalize_value(row['Description'])
            if not description:
                description = 'No description'
            
            country = self.normalize_value(row['Country'])
            if not country:
                country = 'Unknown'
            
            activity = self.normalize_value(row['Activity'])
            if not activity:
                activity = 'No Activity'
            
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
            
            cleaned_rows.append(cleaned_row)
        
        # Create cleaned dataframe
        cleaned_df = pd.DataFrame(cleaned_rows)
        self.stats['final_rows'] = len(cleaned_df)
        
        # Save cleaned data
        cleaned_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        self.log(f"\n[SUCCESS] Cleaned data saved to {output_file}")
        self.log(f"Original rows: {self.stats['original_rows']}")
        self.log(f"Final rows: {self.stats['final_rows']}")
        self.log(f"Removed rows: {self.stats['removed_rows']}")
        self.log(f"Retention rate: {(self.stats['final_rows']/self.stats['original_rows']*100):.2f}%")
        
        return cleaned_df
    
    def generate_quantitative_summary(self, df, output_dir='eda_outputs'):
        """Generate quantitative statistical summaries"""
        os.makedirs(output_dir, exist_ok=True)
        
        summary_file = os.path.join(output_dir, 'quantitative_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("QUANTITATIVE DATA SUMMARY - TASK 2\n")
            f.write("=" * 70 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Total features: {len(df.columns)}\n")
            f.write(f"Features: {', '.join(df.columns)}\n\n")
            
            # Target variable statistics
            f.write("TARGET VARIABLE STATISTICS\n")
            f.write("-" * 70 + "\n")
            
            targets = {
                'Weather': self.VALID_WEATHER,
                'Time of Day': self.VALID_TIME,
                'Season': self.VALID_SEASON,
                'Mood/Emotion': self.VALID_MOOD
            }
            
            for col, valid_options in targets.items():
                f.write(f"\n{col}:\n")
                value_counts = df[col].value_counts()
                f.write(f"  Number of unique classes: {df[col].nunique()}\n")
                f.write(f"  Valid classes: {', '.join(valid_options)}\n")
                f.write(f"  Class distribution:\n")
                for class_name, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    f.write(f"    {class_name}: {count} ({pct:.2f}%)\n")
                
                # Calculate entropy
                p = value_counts / len(df)
                entropy = -(p * np.log2(p + 1e-12)).sum()
                f.write(f"  Shannon Entropy: {entropy:.4f}\n")
                
                # Imbalance ratio
                if len(value_counts) > 1:
                    max_count = value_counts.max()
                    min_count = value_counts.min()
                    imbalance = max_count / min_count
                    f.write(f"  Class Imbalance Ratio: {imbalance:.2f}:1\n")
            
            # Country statistics
            f.write("\n\nCOUNTRY STATISTICS\n")
            f.write("-" * 70 + "\n")
            country_counts = df['Country'].value_counts()
            f.write(f"Number of unique countries: {df['Country'].nunique()}\n")
            f.write(f"Top 15 countries:\n")
            for country, count in country_counts.head(15).items():
                f.write(f"  {country}: {count} ({count/len(df)*100:.2f}%)\n")
            
            # Activity statistics
            f.write("\n\nACTIVITY STATISTICS\n")
            f.write("-" * 70 + "\n")
            activity_counts = df['Activity'].value_counts()
            f.write(f"Number of unique activities: {df['Activity'].nunique()}\n")
            for activity, count in activity_counts.items():
                f.write(f"  {activity}: {count} ({count/len(df)*100:.2f}%)\n")
            
            # Missing data analysis
            f.write("\n\nMISSING DATA ANALYSIS\n")
            f.write("-" * 70 + "\n")
            for col in df.columns:
                missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == '').sum()
                missing_pct = (missing / len(df)) * 100
                f.write(f"{col}: {missing} missing ({missing_pct:.2f}%)\n")
        
        self.log(f"[OK] Saved quantitative summary to {summary_file}")
    
    def generate_visualizations(self, df, output_dir='eda_outputs'):
        """Generate comprehensive EDA visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.log("\n" + "=" * 70)
        self.log("GENERATING EDA VISUALIZATIONS")
        self.log("=" * 70)
        
        # 1. Target variable distributions (histograms)
        self.log("\n1. Generating target variable distributions...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Target Variable Distributions', fontsize=16, fontweight='bold')
        
        targets = ['Weather', 'Time of Day', 'Season', 'Mood/Emotion']
        for idx, col in enumerate(targets):
            ax = axes[idx // 2, idx % 2]
            counts = df[col].value_counts()
            
            bars = ax.bar(range(len(counts)), counts.values, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/target_distributions.png', dpi=300, bbox_inches='tight')
        self.log(f"[OK] Saved: {output_dir}/target_distributions.png")
        plt.close()
        
        # 2. Correlation heatmap
        self.log("\n2. Generating correlation heatmap...")
        from sklearn.preprocessing import LabelEncoder
        
        encoded_df = df[targets].copy()
        for col in targets:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col])
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = encoded_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Between Target Variables', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/label_correlation.png', dpi=300, bbox_inches='tight')
        self.log(f"[OK] Saved: {output_dir}/label_correlation.png")
        plt.close()
        
        # 3. Country distribution (histogram)
        self.log("\n3. Generating country distribution...")
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
        self.log(f"[OK] Saved: {output_dir}/country_distribution.png")
        plt.close()
        
        # 4. Activity distribution (histogram)
        self.log("\n4. Generating activity distribution...")
        plt.figure(figsize=(12, 6))
        activity_counts = df['Activity'].value_counts()
        bars = plt.bar(range(len(activity_counts)), activity_counts.values, color='teal', alpha=0.7)
        plt.xticks(range(len(activity_counts)), activity_counts.index, rotation=45, ha='right')
        plt.xlabel('Activity', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Activity Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/activity_distribution.png', dpi=300, bbox_inches='tight')
        self.log(f"[OK] Saved: {output_dir}/activity_distribution.png")
        plt.close()
        
        # 5. Co-occurrence heatmaps (scatter-like visualization)
        self.log("\n5. Generating co-occurrence heatmaps...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Multi-Label Co-occurrence Patterns', fontsize=16, fontweight='bold')
        
        # Weather vs Time of Day
        ct = pd.crosstab(df['Weather'], df['Time of Day'])
        sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0, 0], cbar_kws={'label': 'Count'})
        axes[0, 0].set_title('Weather vs Time of Day', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time of Day')
        axes[0, 0].set_ylabel('Weather')
        
        # Weather vs Season
        ct = pd.crosstab(df['Weather'], df['Season'])
        sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0, 1], cbar_kws={'label': 'Count'})
        axes[0, 1].set_title('Weather vs Season', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Season')
        axes[0, 1].set_ylabel('Weather')
        
        # Time of Day vs Season
        ct = pd.crosstab(df['Time of Day'], df['Season'])
        sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': 'Count'})
        axes[1, 0].set_title('Time of Day vs Season', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Season')
        axes[1, 0].set_ylabel('Time of Day')
        
        # Mood vs Weather
        ct = pd.crosstab(df['Mood/Emotion'], df['Weather'])
        sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Count'})
        axes[1, 1].set_title('Mood/Emotion vs Weather', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Weather')
        axes[1, 1].set_ylabel('Mood/Emotion')
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cooccurrence_heatmaps.png', dpi=300, bbox_inches='tight')
        self.log(f"[OK] Saved: {output_dir}/cooccurrence_heatmaps.png")
        plt.close()
        
        # 6. Pie charts for class distributions
        self.log("\n6. Generating pie charts...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Class Distribution Pie Charts', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(targets):
            ax = axes[idx // 2, idx % 2]
            value_counts = df[col].value_counts()
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index,
                                             autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pie_charts.png', dpi=300, bbox_inches='tight')
        self.log(f"[OK] Saved: {output_dir}/pie_charts.png")
        plt.close()
        
        self.log("\n[OK] All visualizations generated!")
    
    def print_preprocessing_summary(self):
        """Print preprocessing summary"""
        print("\n" + "=" * 70)
        print("PREPROCESSING SUMMARY")
        print("=" * 70)
        print(f"Original rows: {self.stats['original_rows']}")
        print(f"Final rows: {self.stats['final_rows']}")
        print(f"Removed rows: {self.stats['removed_rows']}")
        print(f"Retention rate: {(self.stats['final_rows']/self.stats['original_rows']*100):.2f}%")
        
        print("\nRemoval Reasons:")
        print("-" * 70)
        for reason, count in self.stats['removal_reasons'].most_common():
            print(f"  {reason}: {count}")
        
        print("\nValidation Statistics:")
        print("-" * 70)
        print(f"  Valid URLs: {self.stats['valid_urls']}")
        print(f"  Invalid URLs: {self.stats['url_validation_failed']}")
        print(f"  Invalid Weather: {self.stats['invalid_weather']}")
        print(f"  Invalid Time of Day: {self.stats['invalid_time']}")
        print(f"  Invalid Season: {self.stats['invalid_season']}")
        print(f"  Invalid Mood/Emotion: {self.stats['invalid_mood']}")
        print("=" * 70)


def main():
    """Main execution function"""
    preprocessor = Task2Preprocessor()
    
    # Preprocess data
    # Note: Set validate_urls=False for faster processing (URLs will still be format-validated)
    # Set validate_urls=True to actually check if URLs are accessible (slower but more thorough)
    cleaned_df = preprocessor.preprocess_data(
        input_file='data.csv',
        output_file='cleaned_data.csv',
        validate_urls=False,  # Set to True to validate URLs (slower)
        verbose=True
    )
    
    if cleaned_df is not None and len(cleaned_df) > 0:
        # Print summary
        preprocessor.print_preprocessing_summary()
        
        # Generate quantitative summary
        preprocessor.generate_quantitative_summary(cleaned_df)
        
        # Generate visualizations
        preprocessor.generate_visualizations(cleaned_df)
        
        print("\n" + "=" * 70)
        print("TASK 2 COMPLETE!")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  - cleaned_data.csv")
        print(f"  - eda_outputs/quantitative_summary.txt")
        print(f"  - eda_outputs/target_distributions.png")
        print(f"  - eda_outputs/label_correlation.png")
        print(f"  - eda_outputs/country_distribution.png")
        print(f"  - eda_outputs/activity_distribution.png")
        print(f"  - eda_outputs/cooccurrence_heatmaps.png")
        print(f"  - eda_outputs/pie_charts.png")
        print("\n" + "=" * 70)
    else:
        print("\nERROR: No data remaining after preprocessing!")


if __name__ == "__main__":
    main()

