import os
import csv
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow/Keras
try:
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.optimizers import Adam
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError as e:
    print(f"Error: {e}")
    print("Installing required packages...")
    os.system("pip install tensorflow scikit-learn -q")
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.optimizers import Adam
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

DATA_FOLDER = r"c:\Users\Siddhartha Reddy\Desktop\radar\20250319"

# Category mapping
CATEGORY_MAP = {
    'exposed': 0,
    'hidden': 1,
    'stock': 2
}

REVERSE_MAP = {v: k for k, v in CATEGORY_MAP.items()}

print("="*100)
print("NEURAL NETWORK TRAINING - CORN COB CLASSIFICATION FROM RADAR DATA")
print("="*100)
print()

# STEP 1: LOAD AND PARSE CSV DATA
print("[STEP 1] Loading and parsing CSV radar data...")
print("-"*100)

def extract_category_from_filename(filename):
    """Extract ground truth category from filename"""
    if 'exposed' in filename:
        return 'exposed'
    elif 'hidden' in filename:
        return 'hidden'
    elif 'stock' in filename:
        return 'stock'
    return None

def load_csv_data(filepath):
    """Load radar signal data from CSV file"""
    scan_data_list = []
    metadata = []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Find the MrmFullScanInfo header row
    scan_info_header_idx = None
    for i, row in enumerate(rows):
        if row and len(row) > 2:
            # Check with stripping whitespace
            if any('MrmFullScanInfo' in col.strip() for col in row) and any('ScanData' in col.strip() for col in row):
                scan_info_header_idx = i
                break
    
    if scan_info_header_idx is None:
        return [], []
    
    # Find index where ScanData starts
    header_row = rows[scan_info_header_idx]
    scan_data_start_idx = None
    for j, col in enumerate(header_row):
        if 'ScanData' in col.strip():
            scan_data_start_idx = j
            break
    
    if scan_data_start_idx is None:
        return [], []
    
    # Extract scan data from rows after the header
    for row_idx in range(scan_info_header_idx + 1, len(rows)):
        row = rows[row_idx]
        
        try:
            # Check if this is a MrmFullScanInfo data row (col 1 should be 'MrmFullScanInfo')
            if row and len(row) > scan_data_start_idx:
                row1_stripped = row[1].strip() if len(row) > 1 else ""
                if row1_stripped == "MrmFullScanInfo":
                    timestamp = float(row[0].strip())
                    
                    # Extract scan data starting from the correct index
                    scan_values = []
                    for i in range(scan_data_start_idx, len(row)):
                        try:
                            val = float(row[i].strip())
                            scan_values.append(val)
                        except (ValueError, TypeError):
                            pass
                    
                    if len(scan_values) > 10:
                        scan_data_list.append(np.array(scan_values))
                        metadata.append({
                            'timestamp': timestamp,
                            'num_samples': len(scan_values)
                        })
        except (ValueError, IndexError):
            pass
    
    return scan_data_list, metadata

# Load all CSV files
all_data = []
all_labels = []
file_analysis = {}

csv_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')])

for csv_file in csv_files:
    filepath = os.path.join(DATA_FOLDER, csv_file)
    category = extract_category_from_filename(csv_file)
    
    if category:
        print(f"  Loading: {csv_file}")
        scan_data, metadata = load_csv_data(filepath)
        
        print(f"    - Records found: {len(scan_data)}")
        if scan_data:
            sample_lens = str([len(s) for s in scan_data[:3]])
            print(f"    - Sample lengths: {sample_lens}")
            print(f"    - Category: {category.upper()}")
            print(f"    - Label value: {CATEGORY_MAP[category]}")
        
        all_data.extend(scan_data)
        all_labels.extend([CATEGORY_MAP[category]] * len(scan_data))
        
        file_analysis[csv_file] = {
            'category': category,
            'records': len(scan_data),
            'sample_size': len(scan_data[0]) if scan_data else 0
        }

print()
print(f"Total records loaded: {len(all_data)}")
print(f"Total labels: {len(all_labels)}")
print()

# STEP 2: PREPROCESS DATA
print("[STEP 2] Preprocessing radar signal data...")
print("-"*100)

if len(all_data) == 0:
    print("ERROR: No scan data found in CSV files!")
    print("Please check CSV format and file paths.")
    exit(1)

# Find common length for padding
max_length = max([len(d) for d in all_data]) if all_data else 0
min_length = min([len(d) for d in all_data]) if all_data else 0

print(f"  Signal length range: {min_length} to {max_length} samples")

# Pad all signals to the same length
def pad_signal(signal, target_length):
    """Pad or truncate signal to target length"""
    if len(signal) < target_length:
        padding = np.zeros(target_length - len(signal))
        return np.concatenate([signal, padding])
    else:
        return signal[:target_length]

X = np.array([pad_signal(d, max_length) for d in all_data])
y = np.array(all_labels)

print(f"  Padded data shape: {X.shape}")
print(f"  Labels shape: {y.shape}")
print()

# Normalize data
print("  Normalizing radar signals...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  Data mean: {X_scaled.mean():.6f}")
print(f"  Data std: {X_scaled.std():.6f}")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print()

# Class distribution
print("Class distribution in full dataset:")
for label, category in REVERSE_MAP.items():
    count = np.sum(y == label)
    pct = 100 * count / len(y)
    print(f"    {category.upper():8} (label {label}): {count:4d} samples ({pct:5.1f}%)")
print()

# STEP 3: BUILD NEURAL NETWORK MODEL
print("[STEP 3] Building neural network model...")
print("-"*100)

# Reshape for Dense layers
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Build model
model = Sequential([
    layers.Dense(256, activation='relu', input_shape=(max_length,)),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model created with 3 output classes (Exposed, Hidden, Stock)")
print()

# STEP 4: TRAIN MODEL
print("[STEP 4] Training neural network...")
print("-"*100)

history = model.fit(
    X_train_reshaped, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

print(f"Training completed!")
print(f"  Final training loss: {history.history['loss'][-1]:.4f}")
print(f"  Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print()

# STEP 5: EVALUATE MODEL
print("[STEP 5] Evaluating model on test set...")
print("-"*100)

y_pred = model.predict(X_test_reshaped, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)

test_accuracy = accuracy_score(y_test, y_pred_labels)
print(f"Test Set Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred_labels, 
                          target_names=[REVERSE_MAP[i] for i in range(3)]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_labels)
print(cm)
print()

# STEP 6: PREDICT ON ALL DATA
print("[STEP 6] Making predictions on all data...")
print("-"*100)

# Predict all data
all_predictions = model.predict(X_scaled, verbose=0)
all_pred_labels = np.argmax(all_predictions, axis=1)
all_pred_confidence = np.max(all_predictions, axis=1)

results = {
    'training_summary': {
        'total_samples': len(all_data),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'model_accuracy': float(test_accuracy),
        'training_date': datetime.now().isoformat(),
        'signal_length': int(max_length)
    },
    'file_analysis': file_analysis,
    'predictions': []
}

# Match predictions back to original CSV files
current_idx = 0
for csv_file in csv_files:
    filepath = os.path.join(DATA_FOLDER, csv_file)
    category = extract_category_from_filename(csv_file)
    
    if category:
        num_records = file_analysis[csv_file]['records']
        if num_records > 0:
            file_preds = all_pred_labels[current_idx:current_idx + num_records]
            file_confs = all_pred_confidence[current_idx:current_idx + num_records]
            
            # Calculate statistics
            pred_counts = np.bincount(file_preds, minlength=3)
            
            file_result = {
                'csv_file': csv_file,
                'ground_truth_category': category,
                'ground_truth_label': CATEGORY_MAP[category],
                'total_records': num_records,
                'predictions': {
                    'exposed_count': int(pred_counts[0]),
                    'hidden_count': int(pred_counts[1]),
                    'stock_count': int(pred_counts[2]),
                    'exposed_percent': float(100 * pred_counts[0] / num_records),
                    'hidden_percent': float(100 * pred_counts[1] / num_records),
                    'stock_percent': float(100 * pred_counts[2] / num_records)
                },
                'average_confidence': float(np.mean(file_confs)),
                'confidence_range': [float(np.min(file_confs)), float(np.max(file_confs))],
                'predicted_majority_class': REVERSE_MAP[np.argmax(pred_counts)],
                'predicted_majority_label': int(np.argmax(pred_counts)),
                'matches_ground_truth': REVERSE_MAP[np.argmax(pred_counts)] == category
            }
            
            results['predictions'].append(file_result)
            current_idx += num_records

# STEP 7: DETAILED OUTPUT
print()
print("="*100)
print("PREDICTION RESULTS BY FILE")
print("="*100)
print()

correct = 0
for result in results['predictions']:
    match_symbol = "[MATCH]" if result['matches_ground_truth'] else "[MISMATCH]"
    
    print(f"{match_symbol} {result['csv_file']}")
    print(f"   Ground Truth: {result['ground_truth_category'].upper()} (label {result['ground_truth_label']})")
    print(f"   Predicted:    {result['predicted_majority_class'].upper()} (label {result['predicted_majority_label']})")
    print(f"   Confidence:   {result['average_confidence']:.4f}")
    print(f"   Distribution:")
    print(f"      - Exposed:  {result['predictions']['exposed_count']:3d} ({result['predictions']['exposed_percent']:5.1f}%)")
    print(f"      - Hidden:   {result['predictions']['hidden_count']:3d} ({result['predictions']['hidden_percent']:5.1f}%)")
    print(f"      - Stock:    {result['predictions']['stock_count']:3d} ({result['predictions']['stock_percent']:5.1f}%)")
    print()
    
    if result['matches_ground_truth']:
        correct += 1

accuracy = correct / len(results['predictions'])
print(f"File-level Accuracy: {correct}/{len(results['predictions'])} ({accuracy*100:.1f}%)")
print()

# SAVE RESULTS
output_json = os.path.join(DATA_FOLDER, 'neural_network_predictions.json')
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

# Save model
model_path = os.path.join(DATA_FOLDER, 'trained_model.h5')
model.save(model_path)

print("="*100)
print(f"OK - Predictions saved to: {output_json}")
print(f"OK - Trained model saved to: {model_path}")
print("="*100)
