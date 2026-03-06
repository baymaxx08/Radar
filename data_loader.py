import pandas as pd
import numpy as np
from pathlib import Path
import json

class RadarDataLoader:
    """Load and preprocess radar scan data from CSV files."""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.scans = []
        self.labels = []
        
    def parse_csv_file(self, filepath, label):
        """
        Parse radar CSV file and extract scan data.
        
        Label mapping:
        0 = exposed_corn_ear (full corn cob - bare)
        1 = hidden_corn_ear (partial corn cob)
        2 = stock (no cob)
        """
        print(f"Loading {filepath.name} with label: {label}")
        
        with open(filepath, 'r') as f:
            scan_index = 0
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Skip header lines
                if line.startswith('Timestamp') and 'Config' in line:
                    continue
                if line.startswith('Timestamp') and 'MrmFullScanInfo' in line:
                    continue
                if line.startswith('Timestamp') and 'MrmDetectionListInfo' in line:
                    continue
                    
                # Parse MrmFullScanInfo line - contains the 480 radar samples
                # Format: timestamp, MrmFullScanInfo, MessageId, SourceId, ..., 480, v1, v2, ..., v480
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
                                scan_array = np.array(values, dtype=np.float32)
                                # Normalize the scan data
                                scan_normalized = self.normalize_scan(scan_array)
                                self.scans.append(scan_normalized)
                                self.labels.append(label)
                                scan_index += 1
                    except Exception as e:
                        # Silently skip lines that can't be parsed
                        pass
        
        print(f"  Loaded {scan_index} scans")
        return scan_index
    
    def normalize_scan(self, scan):
        """Normalize scan data to [-1, 1] range."""
        if len(scan) == 0:
            return scan
        
        scan_min = np.min(scan)
        scan_max = np.max(scan)
        
        if scan_max - scan_min == 0:
            return scan
        
        normalized = 2 * (scan - scan_min) / (scan_max - scan_min) - 1
        return normalized
    
    def load_all_data(self):
        """Load all radar data files."""
        # File to label mapping
        file_labels = {
            'exposed_corn_ear': 0,  # Full corn cob (bare)
            'hidden_corn_ear': 1,   # Partial corn cob
            'stock': 2               # No cob
        }
        
        total_scans = 0
        for filename in self.data_dir.glob('*.csv'):
            for file_type, label in file_labels.items():
                if file_type in filename.name:
                    scans_loaded = self.parse_csv_file(filename, label)
                    total_scans += scans_loaded
                    break
        
        print(f"\nTotal scans loaded: {total_scans}")
        print(f"Class distribution: {np.bincount(self.labels)}")
        
        return np.array(self.scans), np.array(self.labels)
    
    def get_statistics(self):
        """Get statistics about loaded data."""
        if len(self.scans) == 0:
            return None
        
        scans_array = np.array(self.scans)
        labels_array = np.array(self.labels)
        
        stats = {
            'total_samples': len(self.scans),
            'classes': {
                'exposed_corn_ear (full cob)': int(np.sum(labels_array == 0)),
                'hidden_corn_ear (partial cob)': int(np.sum(labels_array == 1)),
                'stock (no cob)': int(np.sum(labels_array == 2))
            },
            'sample_shape': scans_array.shape,
            'scan_length': 480
        }
        return stats


if __name__ == '__main__':
    # Test data loader
    data_dir = Path(r'c:\Users\Siddhartha Reddy\Desktop\radar\20250319')
    loader = RadarDataLoader(data_dir)
    X, y = loader.load_all_data()
    stats = loader.get_statistics()
    print(f"\nData statistics: {json.dumps(stats, indent=2)}")
