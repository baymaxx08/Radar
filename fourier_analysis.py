import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Load the CSV files
exposed_file = r'20250319/202503019_GH_exposed_corn_ear_cls001.csv'
hidden_file = r'20250319/202503019_GH_hidden_corn_ear_cls005.csv'

def extract_scan_data(filepath):
    """Extract ScanData from CSV file"""
    df = pd.read_csv(filepath, skiprows=2)
    
    # Filter for MrmFullScanInfo rows with Filtering=1 (first segment)
    scan_data = df[(df['Timestamp'].notna()) & 
                   (df[' MrmFullScanInfo'].notna()) &
                   (df[' Filtering'] == 1)].copy()
    
    # Extract ScanData column
    scans = []
    for idx, row in scan_data.iterrows():
        # Get all columns after ScanData
        cols = df.columns.tolist()
        scan_start_idx = cols.index(' ScanData')
        row_data = row.iloc[scan_start_idx:].dropna().values
        if len(row_data) > 0:
            scans.append(row_data.astype(float))
    
    return scans

def compute_fft(signal):
    """Compute FFT of a signal"""
    if len(signal) == 0:
        return None, None
    
    # Apply Hamming window to reduce spectral leakage
    windowed = signal * np.hamming(len(signal))
    
    # Compute FFT
    fft_vals = np.abs(fft(windowed))
    freqs = fftfreq(len(signal))
    
    # Keep only positive frequencies
    positive_freq_idx = freqs >= 0
    return freqs[positive_freq_idx], fft_vals[positive_freq_idx]

# Extract scan data
print("Loading exposed corn data...")
exposed_scans = extract_scan_data(exposed_file)
print(f"Found {len(exposed_scans)} exposed scans")

print("Loading hidden corn data...")
hidden_scans = extract_scan_data(hidden_file)
print(f"Found {len(hidden_scans)} hidden scans")

# Compute FFT for first scan of each
if exposed_scans and hidden_scans:
    exp_freq, exp_fft = compute_fft(exposed_scans[0])
    hid_freq, hid_fft = compute_fft(hidden_scans[0])
    
    # Compute statistics
    exp_power = np.sum(exp_fft**2)
    hid_power = np.sum(hid_fft**2)
    
    # Find peak frequencies
    exp_peak_idx = np.argmax(exp_fft[1:50]) + 1  # Skip DC component
    hid_peak_idx = np.argmax(hid_fft[1:50]) + 1
    
    print("\n" + "="*60)
    print("FOURIER TRANSFORM ANALYSIS: EXPOSED vs HIDDEN CORN")
    print("="*60)
    
    print(f"\nEXPOSED CORN:")
    print(f"  Total spectral power: {exp_power:.2e}")
    print(f"  Peak frequency bin: {exp_peak_idx}")
    print(f"  Peak magnitude: {exp_fft[exp_peak_idx]:.2e}")
    
    print(f"\nHIDDEN CORN:")
    print(f"  Total spectral power: {hid_power:.2e}")
    print(f"  Peak frequency bin: {hid_peak_idx}")
    print(f"  Peak magnitude: {hid_fft[hid_peak_idx]:.2e}")
    
    print(f"\nPower Ratio (Hidden/Exposed): {hid_power/exp_power:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time domain comparisons
    axes[0, 0].plot(exposed_scans[0], label='Exposed', alpha=0.7)
    axes[0, 0].set_title('Time Domain: Exposed Corn Signal')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(hidden_scans[0], label='Hidden', color='orange', alpha=0.7)
    axes[0, 1].set_title('Time Domain: Hidden Corn Signal')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # FFT magnitude spectrum
    axes[1, 0].semilogy(exp_freq[:100], exp_fft[:100], label='Exposed', alpha=0.7)
    axes[1, 0].set_title('Frequency Domain: Exposed Corn (dB scale)')
    axes[1, 0].set_xlabel('Frequency (normalized)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(hid_freq[:100], hid_fft[:100], label='Hidden', color='orange', alpha=0.7)
    axes[1, 1].set_title('Frequency Domain: Hidden Corn (dB scale)')
    axes[1, 1].set_xlabel('Frequency (normalized)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fourier_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: fourier_comparison.png")
    
    # Comparison plot
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(exp_freq[:150], exp_fft[:150], label='Exposed Corn', linewidth=2, alpha=0.8)
    ax.semilogy(hid_freq[:150], hid_fft[:150], label='Hidden Corn', linewidth=2, alpha=0.8)
    ax.set_xlabel('Frequency Bin', fontsize=12)
    ax.set_ylabel('Magnitude (log scale)', fontsize=12)
    ax.set_title('FFT Comparison: Exposed vs Hidden Corn Cobs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('fft_overlay_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fft_overlay_comparison.png")
    
    plt.show()
else:
    print("Error: Could not extract scan data from files")
