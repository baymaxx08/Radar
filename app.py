import os
import sys
from pathlib import Path
import numpy as np
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import traceback

# Add the project directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import RadarDataLoader
from model import CornCobClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Initialize global variables
classifier = CornCobClassifier()
model_path = Path(__file__).parent / 'corn_cob_model.h5'
training_data = None
model_trained = False

def train_model():
    """Train the model on available data."""
    global classifier, model_trained, training_data
    
    try:
        print("Loading training data...")
        data_dir = Path(__file__).parent / '20250319'
        loader = RadarDataLoader(data_dir)
        X, y = loader.load_all_data()
        stats = loader.get_statistics()
        
        training_data = {'X': X, 'y': y, 'stats': stats}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build and train model
        classifier.build_model()
        classifier.train(X_train, y_train, epochs=10, batch_size=16)
        
        # Evaluate
        classifier.evaluate(X_test, y_test)
        
        # Save model
        classifier.save_model(str(model_path))
        model_trained = True
        
        print("Model training completed!")
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status."""
    try:
        return jsonify({
            'status': 'running',
            'model_trained': model_trained,
            'model_path': str(model_path) if model_trained else None,
            'training_data': {
                'stats': training_data['stats'] if training_data else None
            } if training_data else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    """Train the model."""
    try:
        success = train_model()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully',
                'stats': training_data['stats']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to train model'
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on radar scan data."""
    try:
        if not model_trained:
            return jsonify({'error': 'Model not trained yet. Train model first.'}), 400
        
        data = request.json
        scans = np.array(data.get('scans', []), dtype=np.float32)
        
        if scans.shape[0] == 0:
            return jsonify({'error': 'No scans provided'}), 400
        
        # Normalize scans
        for i in range(scans.shape[0]):
            if np.max(scans[i]) - np.min(scans[i]) != 0:
                scans[i] = 2 * (scans[i] - np.min(scans[i])) / (np.max(scans[i]) - np.min(scans[i])) - 1
        
        # Make predictions
        predictions = classifier.predict_with_confidence(scans)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'num_predictions': len(predictions)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-csv', methods=['POST'])
def analyze_csv():
    """Analyze CSV file and predict."""
    try:
        if not model_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        # Get CSV file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily and use RadarDataLoader to parse it
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.csv', delete=False) as tmp:
            file.seek(0)
            tmp.write(file.read())
            tmp_path = tmp.name
        
        try:
            # Use RadarDataLoader which handles the CSV format correctly
            loader = RadarDataLoader(Path(tmp_path).parent)
            
            scans = []
            with open(tmp_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('Timestamp'):
                        continue
                        
                    if 'MrmFullScanInfo' in line:
                        try:
                            parts = line.split(',')
                            
                            # Find the index of 480 (number of samples)
                            num_samples_idx = -1
                            for i, part in enumerate(parts):
                                if part.strip() == '480':
                                    num_samples_idx = i
                                    break
                            
                            if num_samples_idx != -1 and num_samples_idx + 480 < len(parts):
                                # Extract the 480 values after the 480 marker
                                values = []
                                for i in range(num_samples_idx + 1, num_samples_idx + 481):
                                    try:
                                        val = int(parts[i].strip())
                                        values.append(val)
                                    except ValueError:
                                        break
                                
                                if len(values) == 480:  # Valid scan with 480 samples
                                    scan = np.array(values, dtype=np.float32)
                                    # Normalize
                                    if np.max(scan) - np.min(scan) != 0:
                                        scan = 2 * (scan - np.min(scan)) / (np.max(scan) - np.min(scan)) - 1
                                    scans.append(scan)
                        except Exception as parse_err:
                            print(f"Error parsing line: {parse_err}")
                            continue
            
            if len(scans) == 0:
                # Provide detailed debug info
                with open(tmp_path, 'r') as f:
                    all_lines = f.readlines()
                    mrm_lines = [l for l in all_lines if 'MrmFullScanInfo' in l]
                    
                print(f"DEBUG: No scans found. File has {len(all_lines)} total lines, {len(mrm_lines)} MrmFullScanInfo lines")
                if mrm_lines:
                    print(f"DEBUG: First MrmFullScanInfo line: {mrm_lines[0][:200]}")
                    parts = mrm_lines[0].split(',')
                    print(f"DEBUG: Line split into {len(parts)} parts")
                    
                return jsonify({
                    'error': 'No valid radar scans found in file',
                    'debug': {
                        'total_lines': len(all_lines),
                        'mrm_lines': len(mrm_lines),
                        'hint': 'File should contain MrmFullScanInfo lines with exactly 480 comma-separated sample values'
                    }
                }), 400
            
            # Make predictions
            X = np.array(scans)
            predictions = classifier.predict_with_confidence(X)
            
            # Aggregate results
            class_counts = {'Full Corn Cob (Bare)': 0, 'Partial Corn Cob (Hidden)': 0, 'No Cob (Stock)': 0}
            avg_probs = {'full_corn_cob': 0, 'partial_corn_cob': 0, 'no_cob': 0}
            
            for pred in predictions:
                class_counts[pred['class']] += 1
                for key, val in pred['all_probabilities'].items():
                    avg_probs[key] += val
            
            for key in avg_probs:
                avg_probs[key] /= len(predictions)
            
            return jsonify({
                'status': 'success',
                'scans_analyzed': len(scans),
                'predictions': predictions,
                'summary': {
                    'class_distribution': class_counts,
                    'average_probabilities': avg_probs
                }
            })
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        print(f"Error in analyze_csv: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Corn Cob Classification Web Server...")
    print("Server running at http://localhost:5000")
    print("Click the 'Train Model' button in the web interface to train the model.")
    
    # Get port from environment variable (for Render.com deployment)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    # Run Flask app WITHOUT auto-training
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
