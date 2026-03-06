# Configuration Settings for Corn Cob Classification System

# Flask Web Server Configuration
FLASK_CONFIG = {
    'HOST': '0.0.0.0',              # Listen on all network interfaces
    'PORT': 5000,                   # Web server port
    'DEBUG': True,                  # Enable debug mode (development only)
    'ALLOWED_UPLOADS': ['.csv'],    # Allowed file types for upload
    'MAX_FILE_SIZE': 50 * 1024 * 1024,  # Max 50MB file uploads
}

# Neural Network Model Configuration
MODEL_CONFIG = {
    'INPUT_SHAPE': 480,             # Radar scan length (480 samples)
    'NUM_CLASSES': 3,               # Three classification classes
    'MODEL_PATH': 'corn_cob_model.h5',  # Where to save trained model
    'BATCH_SIZE': 16,               # Training batch size
    'EPOCHS': 30,                   # Number of training epochs
    'VALIDATION_SPLIT': 0.2,        # 20% validation data
    'LEARNING_RATE': 0.001,         # Adam optimizer learning rate
}

# Data Configuration
DATA_CONFIG = {
    'DATA_DIR': '20250319',         # Directory with CSV files
    'CLASS_NAMES': {
        0: 'Full Corn Cob (Bare)',
        1: 'Partial Corn Cob (Hidden)',
        2: 'No Cob (Stock)'
    },
    'FILE_PATTERNS': {
        'exposed_corn_ear': 0,       # Class 0
        'hidden_corn_ear': 1,        # Class 1
        'stock': 2                   # Class 2
    }
}

# Data Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'NORMALIZATION_TYPE': 'minmax',  # 'minmax' or 'zscore'
    'MINMAX_RANGE': [-1, 1],        # Normalize to [-1, 1]
    'SCAN_LENGTH': 480,             # Expected scan length
    'REMOVE_OUTLIERS': False,       # Remove extreme values
    'OUTLIER_THRESHOLD': 3.0,       # Standard deviations for outlier detection
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.5,    # Minimum confidence for prediction
    'RETURN_TOP_K': 3,              # Return top K predictions
    'BATCH_PREDICTION_SIZE': 32,    # Batch size for predictions
}

# Logging Configuration
LOGGING_CONFIG = {
    'LEVEL': 'INFO',                # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'LOG_FILE': 'app.log',          # Log file path
    'MAX_LOG_SIZE': 10 * 1024 * 1024,  # Max 10MB per log file
}

# Training Configuration
TRAINING_CONFIG = {
    'AUTO_TRAIN_ON_STARTUP': True,  # Automatically train model on app start
    'TRAIN_TEST_SPLIT': 0.2,        # 80% training, 20% testing
    'RANDOM_SEED': 42,              # For reproducibility
    'SHUFFLE_DATA': True,           # Shuffle training data
    'STRATIFIED_SPLIT': True,       # Maintain class distribution in splits
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'USE_GPU': True,                # Try to use GPU if available
    'MAX_WORKERS': 4,               # Number of workers for data loading
    'PREDICTION_TIMEOUT': 30,       # Seconds before prediction times out
}

if __name__ == '__main__':
    print("Configuration loaded successfully")
    print(f"Flask will listen on http://{FLASK_CONFIG['HOST']}:{FLASK_CONFIG['PORT']}")
    print(f"Model will be loaded from: {MODEL_CONFIG['MODEL_PATH']}")
    print(f"Data directory: {DATA_CONFIG['DATA_DIR']}")
