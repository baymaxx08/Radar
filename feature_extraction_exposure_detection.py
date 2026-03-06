"""
Feature Extraction & Exposure Detection (Full vs. Partial)
=========================================================
Extract signal features from radar waveforms and classify exposure levels
using Random Forest ML model.

This classifies corn cob exposure: Full Exposure vs. Partial Exposure
"""

import numpy as np
import pandas as pd
import csv
import os
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: Feature Extraction Functions
# =============================================================================

def extract_signal_energy(scan):
    """Calculate total signal energy (sum of absolute values)"""
    return np.sum(np.abs(scan))

def extract_peak_strength(scan):
    """Get maximum amplitude from the signal"""
    return np.max(np.abs(scan))

def extract_strong_peaks(scan, threshold=10000):
    """Count reflections stronger than threshold"""
    return np.sum(np.abs(scan) > threshold)

def extract_variance(scan):
    """Calculate signal variance (stability measure)"""
    return np.var(scan)

def extract_mean_amplitude(scan):
    """Calculate mean absolute amplitude"""
    return np.mean(np.abs(scan))

def extract_std_amplitude(scan):
    """Calculate standard deviation of amplitude"""
    return np.std(np.abs(scan))

def extract_peak_to_mean_ratio(scan):
    """Ratio of peak to mean (higher = more concentrated energy)"""
    mean_val = np.mean(np.abs(scan))
    if mean_val == 0:
        return 0
    return np.max(np.abs(scan)) / mean_val

def extract_signal_range(scan):
    """Dynamic range (max - min)"""
    return np.max(scan) - np.min(scan)

# =============================================================================
# STEP 2: Load CSV Data & Extract Features
# =============================================================================

def load_csv_data_with_features(csv_dir):
    """Load CSV files and extract all features from each scan"""
    
    print("[STEP 1] Loading CSV files and extracting features...")
    print("=" * 70)
    
    features_list = []
    file_mapping = {}
    
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    
    for file_idx, csv_file in enumerate(csv_files):
        csv_path = os.path.join(csv_dir, csv_file)
        
        # Determine category from filename
        if "exposed" in csv_file.lower():
            category = "Exposed"
        elif "hidden" in csv_file.lower():
            category = "Hidden"
        elif "stock" in csv_file.lower():
            category = "Stock"
        else:
            category = "Unknown"
        
        # Map for later reference
        if category not in file_mapping:
            file_mapping[category] = []
        file_mapping[category].append(csv_file)
        
        scans_in_file = 0
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Find header row
            header_row = None
            data_start = 0
            for idx, row in enumerate(rows):
                if len(row) > 0 and 'MrmFullScanInfo' in row[1]:
                    header_row = row
                    data_start = idx + 1
                    break
            
            if header_row is None:
                print(f"  ⚠ {csv_file}: Header not found, skipping")
                continue
            
            # Find ScanData column index (column 16 is ScanData)
            scan_data_idx = 16
            
            # Extract scans from data rows
            for row_idx in range(data_start, len(rows)):
                row = rows[row_idx]
                
                if len(row) <= scan_data_idx:
                    continue
                
                # Extract ScanData values (all columns from index 16 onwards)
                try:
                    scan_values = [float(val.strip()) for val in row[scan_data_idx:] if val.strip()]
                    
                    if len(scan_values) == 0:
                        continue
                    
                    # Pad or truncate to 480 samples
                    if len(scan_values) < 480:
                        scan_values = scan_values + [0] * (480 - len(scan_values))
                    else:
                        scan_values = scan_values[:480]
                    
                    scan_array = np.array(scan_values)
                    
                    # Extract all features
                    features = {
                        'file': csv_file,
                        'category': category,
                        'energy': extract_signal_energy(scan_array),
                        'peak_strength': extract_peak_strength(scan_array),
                        'strong_peaks': extract_strong_peaks(scan_array, threshold=10000),
                        'variance': extract_variance(scan_array),
                        'mean_amplitude': extract_mean_amplitude(scan_array),
                        'std_amplitude': extract_std_amplitude(scan_array),
                        'peak_to_mean_ratio': extract_peak_to_mean_ratio(scan_array),
                        'signal_range': extract_signal_range(scan_array),
                    }
                    
                    features_list.append(features)
                    scans_in_file += 1
                
                except ValueError:
                    continue
        
        print(f"  ✓ {csv_file} ({category}): {scans_in_file} scans extracted")
    
    print("=" * 70)
    return pd.DataFrame(features_list), file_mapping


# =============================================================================
# STEP 3: Simple Decision Rule (Baseline)
# =============================================================================

def classify_with_decision_rule(energy, strong_peaks):
    """
    Simple decision rule:
    High energy + Many strong peaks → Full exposure
    Otherwise → Partial exposure
    """
    if energy > 2e6 and strong_peaks > 40:
        return "Full Exposure"
    else:
        return "Partial Exposure"

# =============================================================================
# STEP 4: Train Random Forest (RECOMMENDED METHOD)
# =============================================================================

def train_random_forest_model(df):
    """Train Random Forest to classify Full vs. Partial exposure"""
    
    print("\n[STEP 2] Training Random Forest Model...")
    print("=" * 70)
    
    # For exposed files: full exposure
    # For hidden/stock files: partial exposure (less direct signal)
    df['exposure_level'] = df['category'].apply(
        lambda x: "Full Exposure" if x == "Exposed" else "Partial Exposure"
    )
    
    print(f"Total samples: {len(df)}")
    print(f"Full Exposure: {sum(df['exposure_level'] == 'Full Exposure')} samples")
    print(f"Partial Exposure: {sum(df['exposure_level'] == 'Partial Exposure')} samples")
    
    # Feature selection
    feature_cols = ['energy', 'peak_strength', 'strong_peaks', 'variance', 
                   'mean_amplitude', 'std_amplitude', 'peak_to_mean_ratio', 'signal_range']
    
    X = df[feature_cols].values
    y = (df['exposure_level'] == 'Full Exposure').astype(int).values  # 1=Full, 0=Partial
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Model trained successfully")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Partial Exposure', 'Full Exposure']))
    
    print("\nFeature Importance (Top Features):")
    feature_importance = sorted(zip(feature_cols, model.feature_importances_), 
                               key=lambda x: x[1], reverse=True)
    for feat, importance in feature_importance[:5]:
        print(f"  {feat}: {importance*100:.2f}%")
    
    print("=" * 70)
    
    return model, scaler, feature_cols, df

# =============================================================================
# STEP 5: Make Predictions on All Data
# =============================================================================

def predict_exposure_levels(model, scaler, feature_cols, df):
    """Predict exposure level for all samples using trained model"""
    
    print("\n[STEP 3] Making Predictions on All Samples...")
    print("=" * 70)
    
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    df['predicted_exposure'] = ['Full Exposure' if pred == 1 else 'Partial Exposure' 
                               for pred in predictions]
    df['full_exposure_probability'] = probabilities[:, 1]
    df['partial_exposure_probability'] = probabilities[:, 0]
    
    # Accuracy
    correct = (df['predicted_exposure'] == df['exposure_level']).sum()
    accuracy = correct / len(df) * 100
    
    print(f"Predictions completed")
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{len(df)} correct)")
    
    # Per-file accuracy
    print("\nPer-File Accuracy:")
    print("-" * 70)
    
    file_results = {}
    for file_name in df['file'].unique():
        file_df = df[df['file'] == file_name]
        pred_label = file_df['predicted_exposure'].mode()[0]
        true_label = file_df['exposure_level'].iloc[0]
        match = "✓ MATCH" if pred_label == true_label else "✗ MISMATCH"
        
        file_results[file_name] = {
            'predicted_exposure': pred_label,
            'true_exposure': true_label,
            'num_samples': len(file_df),
            'match': match
        }
        
        print(f"  {file_name}")
        print(f"    → Predicted: {pred_label} | True: {true_label} {match}")
        print(f"    → {len(file_df)} samples")
    
    print("=" * 70)
    
    return df, file_results

# =============================================================================
# STEP 6: Generate Reports & Visualizations
# =============================================================================

def generate_feature_statistics(df):
    """Print feature statistics grouped by exposure level"""
    
    print("\n[STEP 4] Feature Statistics by Exposure Level")
    print("=" * 70)
    
    feature_cols = ['energy', 'peak_strength', 'strong_peaks', 'variance', 
                   'mean_amplitude', 'std_amplitude', 'peak_to_mean_ratio', 'signal_range']
    
    for exposure in ['Full Exposure', 'Partial Exposure']:
        print(f"\n{exposure}:")
        print("-" * 70)
        subset = df[df['exposure_level'] == exposure][feature_cols]
        stats = subset.describe()
        print(stats.round(2))

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    
    print("\n" + "=" * 70)
    print("EXPOSURE LEVEL DETECTION - FEATURE EXTRACTION & ML CLASSIFICATION")
    print("=" * 70)
    
    # Define data paths
    csv_dir = r'c:\Users\Siddhartha Reddy\Desktop\radar\20250319'
    
    # Step 1: Load and extract features
    df, file_mapping = load_csv_data_with_features(csv_dir)
    
    if len(df) == 0:
        print("ERROR: No data loaded")
        exit(1)
    
    # Step 2: Train Random Forest model
    model, scaler, feature_cols, df = train_random_forest_model(df)
    
    # Step 3: Make predictions
    df_predictions, file_results = predict_exposure_levels(model, scaler, feature_cols, df)
    
    # Step 4: Display feature statistics
    generate_feature_statistics(df_predictions)
    
    # Step 5: Save results to JSON
    print("\n[STEP 5] Saving Results...")
    print("=" * 70)
    
    output_data = {
        'summary': {
            'total_samples': len(df_predictions),
            'full_exposure_count': sum(df_predictions['predicted_exposure'] == 'Full Exposure'),
            'partial_exposure_count': sum(df_predictions['predicted_exposure'] == 'Partial Exposure'),
            'overall_accuracy_percent': (df_predictions['predicted_exposure'] == df_predictions['exposure_level']).sum() / len(df_predictions) * 100
        },
        'file_predictions': file_results,
        'top_features': {
            'feature': 'energy',
            'interpretation': 'High energy = Full Exposure, Low energy = Partial Exposure'
        }
    }
    
    with open('exposure_level_predictions.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    # Save detailed CSV
    output_df = df_predictions[['file', 'category', 'exposure_level', 'predicted_exposure', 
                                'energy', 'peak_strength', 'strong_peaks', 'variance',
                                'full_exposure_probability', 'partial_exposure_probability']]
    output_df.to_csv('exposure_level_detailed_predictions.csv', index=False)
    
    print(f"  ✓ Results saved to 'exposure_level_predictions.json'")
    print(f"  ✓ Detailed predictions saved to 'exposure_level_detailed_predictions.csv'")
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    
    # Print summary table
    print("\n📊 SUMMARY TABLE")
    print("=" * 70)
    summary_df = output_df.groupby(['file', 'exposure_level']).size().unstack(fill_value=0)
    print(summary_df)
    print("\nKey Statistics:")
    print(f"  • Full Exposure samples: {output_data['summary']['full_exposure_count']}")
    print(f"  • Partial Exposure samples: {output_data['summary']['partial_exposure_count']}")
    print(f"  • Classification Accuracy: {output_data['summary']['overall_accuracy_percent']:.2f}%")
