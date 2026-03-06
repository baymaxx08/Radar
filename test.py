#!/usr/bin/env python
"""
Test and validation script for Corn Cob Classification System
Run this to verify data loading and model training without starting the web server
"""

import sys
from pathlib import Path
import numpy as np

def test_data_loading():
    """Test if data can be loaded successfully."""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    try:
        from data_loader import RadarDataLoader
        print("✓ data_loader module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import data_loader: {e}")
        return False
    
    try:
        data_dir = Path(__file__).parent / '20250319'
        if not data_dir.exists():
            print(f"✗ Data directory not found: {data_dir}")
            return False
        
        print(f"✓ Data directory found: {data_dir}")
        
        loader = RadarDataLoader(data_dir)
        X, y = loader.load_all_data()
        
        print(f"✓ Data loaded successfully")
        print(f"  - Total samples: {len(X)}")
        print(f"  - Sample shape: {X.shape}")
        print(f"  - Class distribution: {np.bincount(y)}")
        
        stats = loader.get_statistics()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            if key != 'classes':
                print(f"  - {key}: {value}")
        
        if stats:
            print(f"\n  Class Distribution:")
            for class_name, count in stats['classes'].items():
                print(f"    - {class_name}: {count} samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_building():
    """Test if model can be built successfully."""
    print("\n" + "="*60)
    print("TEST 2: Model Building")
    print("="*60)
    
    try:
        from model import CornCobClassifier
        print("✓ model module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import model: {e}")
        return False
    
    try:
        classifier = CornCobClassifier()
        classifier.build_model()
        print("✓ Model built successfully")
        
        print("\nModel Summary:")
        summary = classifier.get_summary()
        print(summary)
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction():
    """Test if model can make predictions."""
    print("\n" + "="*60)
    print("TEST 3: Prediction")
    print("="*60)
    
    try:
        from model import CornCobClassifier
        import numpy as np
        
        classifier = CornCobClassifier()
        classifier.build_model()
        
        # Create dummy test data
        X_test = np.random.randn(5, 480).astype(np.float32)
        
        predictions = classifier.predict(X_test)
        print(f"✓ Model can make predictions")
        print(f"  - Predictions shape: {predictions.shape}")
        print(f"  - Sample prediction shape: {predictions[0].shape}")
        
        # Test with confidence
        results = classifier.predict_with_confidence(X_test)
        print(f"✓ Model returns predictions with confidence")
        print(f"  - Number of results: {len(results)}")
        print(f"\nSample prediction:")
        print(f"  - Class: {results[0]['class']}")
        print(f"  - Confidence: {results[0]['confidence']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flask_app():
    """Test if Flask app can be imported."""
    print("\n" + "="*60)
    print("TEST 4: Flask Application")
    print("="*60)
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except Exception as e:
        print(f"✗ Flask not installed: {e}")
        print("  Run: pip install flask flask-cors")
        return False
    
    try:
        # Just check if app.py can be imported without running it
        app_path = Path(__file__).parent / 'app.py'
        if not app_path.exists():
            print(f"✗ app.py not found: {app_path}")
            return False
        
        print(f"✓ app.py found at: {app_path}")
        
        # Check for key components
        with open(app_path, 'r') as f:
            content = f.read()
            required = ['Flask', 'train_model', 'predict', 'analyze_csv']
            for component in required:
                if component in content:
                    print(f"✓ Found {component} function/class")
                else:
                    print(f"✗ Missing {component} function/class")
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking Flask app: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("Corn Cob Classification System - Validation Tests")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Data Loading", test_data_loading()))
    
    if results[0][1]:  # Only proceed if data loading works
        print("\n" + "-"*60)
        results.append(("Model Building", test_model_building()))
        
        if results[1][1]:  # Only proceed if model building works
            print("\n" + "-"*60)
            results.append(("Prediction", test_prediction()))
    
    print("\n" + "-"*60)
    results.append(("Flask Application", test_flask_app()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print("\n" + "-"*60)
    print(f"Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✓ All tests passed! You can now run the application with:")
        print("  python app.py")
        print("  or on Windows:")
        print("  run.bat")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
