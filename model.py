import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle
from pathlib import Path

class SignalAnalyzer:
    """Analyze radar signal features for interpretable classification."""
    
    @staticmethod
    def extract_features(scan):
        """
        Extract interpretable features from 480-sample radar waveform.
        
        Features:
        - energy: Sum of squared amplitudes (signal power)
        - variance: Signal fluctuation/noise level
        - peak_count: Number of prominent spikes
        - peak_strength: Average amplitude of peaks
        - signal_stability: Ratio of strong to weak samples
        
        Note: Input scan is already normalized to [-1, 1] range.
        For normalized data, we scale up by 2.71e8 to match raw signal energy magnitude.
        """
        scan = np.array(scan, dtype=np.float32)
        
        # 1. Energy: Sum of squared values (total signal power)
        # For normalized data [-1,1], multiply by 2.71e8 to recover original magnitude
        energy = np.sum(scan ** 2) * 2.71e8
        
        # 2. Variance: Signal fluctuation (stable=low, noisy=high)
        variance = np.var(scan)
        
        # 3. Peak detection: Find spikes above 1.5σ threshold
        mean = np.mean(scan)
        std = np.std(scan)
        threshold = mean + 1.5 * std
        peaks = scan[scan > threshold]
        peak_count = len(peaks)
        peak_strength = np.mean(np.abs(peaks)) if len(peaks) > 0 else 0
        
        # 4. Signal stability: Percentage of samples above mean
        strong_samples = np.sum(scan > mean)
        signal_stability = strong_samples / len(scan)
        
        # 5. Max amplitude: Strongest reflection
        max_amplitude = np.max(np.abs(scan))
        
        return {
            'energy': float(energy),
            'variance': float(variance),
            'peak_count': int(peak_count),
            'peak_strength': float(peak_strength),
            'signal_stability': float(signal_stability),
            'max_amplitude': float(max_amplitude),
            'mean': float(mean),
            'std': float(std)
        }
    
    @staticmethod
    def classify_by_features(features):
        """
        Simple decision rule classifier based on signal characteristics.
        
        Logic:
        - FULL COB: High energy + strong peaks + stable signal
        - PARTIAL COB: Medium energy + moderate peaks + some noise
        - NO COB: Low energy + few peaks + noisy
        """
        energy = features['energy']
        variance = features['variance']
        peak_count = features['peak_count']
        peak_strength = features['peak_strength']
        signal_stability = features['signal_stability']
        
        # Normalize features for comparison
        norm_energy = energy / 1e6  # Scale to millions
        norm_variance = variance / 1e4  # Scale variance
        
        reasons = []
        scores = {'full': 0, 'partial': 0, 'no_cob': 0}
        
        # Decision rules
        # Note: Energy is similar across all classes, so use peak count and stability instead
        
        # Peak count is more discriminative
        if peak_count > 50:
            scores['full'] += 3
            reasons.append(f"[YES] Many peaks ({peak_count}) - dense structure")
        elif peak_count > 25:
            scores['partial'] += 2
            reasons.append(f"[MAYBE] Moderate peaks ({peak_count}) - partial object")
        else:
            scores['no_cob'] += 3
            reasons.append(f"[NO] Few peaks ({peak_count}) - sparse signal")
        
        # Signal stability (proportion of strong samples)
        if signal_stability > 0.55:
            scores['full'] += 2
            reasons.append(f"[YES] Stable signal ({signal_stability:.1%} strong) - consistent output")
        elif signal_stability > 0.45:
            scores['partial'] += 1
            reasons.append(f"[MAYBE] Moderate stability ({signal_stability:.1%})")
        else:
            scores['no_cob'] += 2
            reasons.append(f"[NO] Unstable signal ({signal_stability:.1%}) - noisy")
        
        # Peak strength (amplitude of reflections)
        if peak_strength > 0.5:
            scores['full'] += 2
            reasons.append(f"[YES] Strong peaks ({peak_strength:.2f} avg) - solid reflection")
        elif peak_strength > 0.2:
            scores['partial'] += 2
            reasons.append(f"[MAYBE] Moderate peaks ({peak_strength:.2f})")
        else:
            scores['no_cob'] += 1
            reasons.append(f"[NO] Weak peaks ({peak_strength:.2f}) - poor reflection")
        
        # Variance (signal noise level) - lower variance = better defined object
        if variance < 0.2:
            scores['full'] += 2
            reasons.append(f"[YES] Low variance ({variance:.3f}) - clean signal")
        elif variance < 0.4:
            scores['partial'] += 1
            reasons.append(f"[MAYBE] Medium variance ({variance:.3f})")
        else:
            scores['no_cob'] += 1
            reasons.append(f"[NO] High variance ({variance:.3f}) - noisy")
        
        # Determine class by highest score
        classification = max(scores, key=scores.get)
        class_map = {'full': 0, 'partial': 1, 'no_cob': 2}
        
        return {
            'class_idx': class_map[classification],
            'class_name': {0: 'Full Corn Cob (Bare)', 1: 'Partial Corn Cob (Hidden)', 2: 'No Cob (Stock)'}[class_map[classification]],
            'scores': scores,
            'reasons': reasons,
            'features': features
        }

class CornCobClassifier:
    """Neural network model for corn cob classification."""
    
    def __init__(self, input_shape=480, num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = ['Full Corn Cob (Bare)', 'Partial Corn Cob (Hidden)', 'No Cob (Stock)']
    
    def build_model(self):
        """Build neural network architecture."""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.input_shape,)),
            
            # Reshape for 1D convolution
            layers.Reshape((self.input_shape, 1)),
            
            # 1D Convolutional blocks
            layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(256, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Global pooling
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(self.model.summary())
        return model
    
    def train(self, X_train, y_train, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model."""
        if self.model is None:
            self.build_model()
        
        print("Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def predict_with_confidence(self, X):
        """
        Make predictions with neural network AND feature-based explanation.
        
        Returns both NN prediction and explainable feature analysis.
        """
        predictions = self.predict(X)
        class_indices = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        class_labels = [self.class_names[idx] for idx in class_indices]
        
        results = []
        for i, (label, conf) in enumerate(zip(class_labels, confidences)):
            # Get feature analysis for this scan
            features = SignalAnalyzer.extract_features(X[i])
            feature_analysis = SignalAnalyzer.classify_by_features(features)
            
            results.append({
                'class': label,
                'confidence': float(conf),
                'class_idx': int(class_indices[i]),
                'all_probabilities': {
                    'full_corn_cob': float(predictions[i][0]),
                    'partial_corn_cob': float(predictions[i][1]),
                    'no_cob': float(predictions[i][2])
                },
                # Feature-based explanation
                'explanation': {
                    'method': 'Signal Feature Analysis',
                    'predicted_class': feature_analysis['class_name'],
                    'confidence_reasons': feature_analysis['reasons'],
                    'feature_scores': feature_analysis['scores'],
                    'signal_metrics': {
                        'energy': f"{features['energy']/1e6:.2f}M",
                        'variance': f"{features['variance']:.0f}",
                        'peaks': features['peak_count'],
                        'stability': f"{features['signal_stability']:.1%}",
                        'max_amplitude': f"{features['max_amplitude']:.0f}"
                    }
                }
            })
        
        return results
    
    def save_model(self, filepath):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def get_summary(self):
        """Get model summary as string."""
        if self.model is None:
            return "Model not built yet."
        
        from io import StringIO
        import sys
        
        buffer = StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        return buffer.getvalue()


if __name__ == '__main__':
    # Test model creation
    classifier = CornCobClassifier()
    classifier.build_model()
    
    # Test with dummy data
    X_dummy = np.random.randn(10, 480).astype(np.float32)
    y_dummy = np.random.randint(0, 3, 10)
    
    classifier.train(X_dummy, y_dummy, epochs=5)
