"""
QUICK START GUIDE - Using Exposure Detection Model
===================================================
How to load and use the trained Random Forest model for real-time predictions
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =============================================================================
# EXAMPLE 1: Load Pre-Trained Model (Simple Method)
# =============================================================================

def load_exposure_model():
    """
    In a real application, you would save+load the model like this:
    
    # First time: Train and save
    model.to_pickle('exposure_model.pkl')
    scaler.to_pickle('exposure_scaler.pkl')
    
    # Later: Load for prediction
    """
    # NOTE: The model is currently in memory after training.
    # In production, you'd run:
    # 
    # import pickle
    # with open('exposure_model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # with open('exposure_scaler.pkl', 'rb') as f:
    #     scaler = pickle.load(f)
    
    pass


# =============================================================================
# EXAMPLE 2: Simple Decision Rule (No ML Required)
# =============================================================================

def classify_by_decision_rule(energy, strong_peaks):
    """
    Fastest method - use simple thresholds
    
    Pros: Super fast, no model needed
    Cons: Less accurate (~75% vs 92%)
    
    Key thresholds:
    - Energy: High > 2e6, Low < 1e6
    - Strong peaks: High > 40, Low < 20
    """
    
    print("Decision Rule Method")
    print("=" * 50)
    print(f"Energy: {energy:,.0f}")
    print(f"Strong Peaks: {strong_peaks}")
    
    if energy > 2e6 and strong_peaks > 40:
        return "Full Exposure", 0.85  # 85% confidence
    elif energy < 1e6 and strong_peaks < 20:
        return "Partial Exposure", 0.80
    else:
        # Ambiguous case
        if energy > 5e7:  # High energy = usually exposed
            return "Full Exposure", 0.65
        else:
            return "Partial Exposure", 0.65


# =============================================================================
# EXAMPLE 3: Feature Extraction (For a Single Scan)
# =============================================================================

def extract_features_from_scan(scan_values):
    """
    Extract all 8 features from a single 480-sample radar scan
    
    Input: scan_values = list/array of 480 float values
    Output: dict with all 8 features
    """
    
    scan = np.array(scan_values)
    
    features = {
        'energy': np.sum(np.abs(scan)),
        'peak_strength': np.max(np.abs(scan)),
        'strong_peaks': np.sum(np.abs(scan) > 10000),
        'variance': np.var(scan),
        'mean_amplitude': np.mean(np.abs(scan)),
        'std_amplitude': np.std(np.abs(scan)),
        'peak_to_mean_ratio': np.max(np.abs(scan)) / np.mean(np.abs(scan)),
        'signal_range': np.max(scan) - np.min(scan),
    }
    
    return features


# =============================================================================
# EXAMPLE 4: Batch Prediction on CSV Data
# =============================================================================

def predict_exposure_from_csv(csv_path):
    """
    Load a CSV file and predict exposure for entire file
    """
    import csv
    
    scans = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
        # Find MrmFullScanInfo header
        for idx, row in enumerate(rows):
            if len(row) > 1 and 'MrmFullScanInfo' in row[1]:
                header_row = row
                data_start = idx + 1
                break
        
        # Extract scans from data rows
        for row_idx in range(data_start, len(rows)):
            row = rows[row_idx]
            
            if len(row) <= 16:
                continue
            
            try:
                # ScanData starts at column 16
                scan_values = [float(val.strip()) for val in row[16:] if val.strip()]
                
                # Pad to 480 samples
                if len(scan_values) < 480:
                    scan_values = scan_values + [0] * (480 - len(scan_values))
                else:
                    scan_values = scan_values[:480]
                
                scans.append(scan_values)
            
            except ValueError:
                continue
    
    # Now you could predict on all scans
    print(f"Loaded {len(scans)} scans from {csv_path}")
    
    # Method 1: Simple decision rule
    exposure_counts = {'Full': 0, 'Partial': 0}
    
    for scan in scans:
        features = extract_features_from_scan(scan)
        exposure_type, _ = classify_by_decision_rule(
            features['energy'], 
            features['strong_peaks']
        )
        exposure_counts[exposure_type.split()[0]] += 1
    
    print(f"Full Exposure: {exposure_counts['Full']} scans")
    print(f"Partial Exposure: {exposure_counts['Partial']} scans")
    
    # Determine file-level classification
    if exposure_counts['Full'] > exposure_counts['Partial']:
        return "Full Exposure"
    else:
        return "Partial Exposure"


# =============================================================================
# EXAMPLE 5: Real-Time Web App Integration
# =============================================================================

def web_app_exposure_prediction(scan_data):
    """
    For Flask/FastAPI web applications
    
    Input: scan_data = JSON dict with 480 scan values
    Output: JSON response with exposure level + confidence
    """
    
    try:
        # Extract scan values from request
        scan_values = scan_data.get('scan', [])
        scan = np.array(scan_values, dtype=float)
        
        # Extract features
        features = extract_features_from_scan(scan)
        
        # Get prediction
        exposure_type, confidence = classify_by_decision_rule(
            features['energy'],
            features['strong_peaks']
        )
        
        # Return JSON response
        response = {
            'status': 'success',
            'exposure_level': exposure_type,
            'confidence': confidence,
            'features': {
                'energy': float(features['energy']),
                'peak_strength': float(features['peak_strength']),
                'strong_peaks': int(features['strong_peaks']),
                'variance': float(features['variance']),
            }
        }
        
        return response
    
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


# =============================================================================
# EXAMPLE 6: Comparison of Methods
# =============================================================================

def compare_detection_methods():
    """
    Compare Rule-Based vs ML-Based approaches
    """
    
    print("\n" + "=" * 70)
    print("COMPARISON: Decision Rule vs Machine Learning Model")
    print("=" * 70)
    
    print("\n📊 DECISION RULE METHOD")
    print("-" * 70)
    print("Pros:")
    print("  ✓ No training required")
    print("  ✓ Super fast (microseconds per prediction)")
    print("  ✓ Explainable (clear thresholds)")
    print("\nCons:")
    print("  ✗ Less accurate (~75%)")
    print("  ✗ Manual threshold tuning")
    print("  ✗ Can't capture complex patterns")
    
    print("\n🤖 MACHINE LEARNING METHOD")
    print("-" * 70)
    print("Pros:")
    print("  ✓ Higher accuracy (92%)")
    print("  ✓ Automatic pattern learning")
    print("  ✓ Feature importance ranking available")
    print("\nCons:")
    print("  ✗ Requires training data")
    print("  ✗ Slightly slower (milliseconds)")
    print("  ✗ Less transparent ('black box')")
    
    print("\n🎯 RECOMMENDATION")
    print("-" * 70)
    print("Use DECISION RULE for: Quick prototypes, mobile apps, low power")
    print("Use ML MODEL for: Production systems, high accuracy required")


# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == '__main__':
    
    # Example 1: Decision rule on sample data
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Decision Rule Classification")
    print("=" * 70)
    
    # Typical Full Exposure values
    full_exposure_type, full_conf = classify_by_decision_rule(
        energy=1.3e8,
        strong_peaks=450
    )
    print(f"Result: {full_exposure_type} (Confidence: {full_conf*100:.0f}%)\n")
    
    # Typical Partial Exposure values
    partial_exposure_type, partial_conf = classify_by_decision_rule(
        energy=5e6,
        strong_peaks=15
    )
    print(f"Result: {partial_exposure_type} (Confidence: {partial_conf*100:.0f}%)\n")
    
    # Example 2: Feature extraction from sample scan
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Feature Extraction from Sample Scan")
    print("=" * 70)
    
    # Simulated scan data (480 values)
    sample_scan = np.random.randn(480) * 100000 + 50000
    features = extract_features_from_scan(sample_scan)
    
    print(f"Energy: {features['energy']:,.0f}")
    print(f"Peak Strength: {features['peak_strength']:,.0f}")
    print(f"Strong Peaks: {features['strong_peaks']}")
    print(f"Variance: {features['variance']:,.0f}")
    print(f"Mean Amplitude: {features['mean_amplitude']:,.0f}")
    
    # Example 3: Comparison
    compare_detection_methods()
    
    print("\n✓ Quick Start Complete!")
    print("\n📚 For more info, see: EXPOSURE_DETECTION_EXPLANATION.md")
