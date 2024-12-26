import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pathlib import Path

def load_csi_sample(real_path, imag_path):
    """Load and combine real and imaginary parts of CSI data."""
    real_data = scipy.io.loadmat(real_path)['myData']
    imag_data = scipy.io.loadmat(imag_path)['myData']
    return real_data + 1j * imag_data

def analyze_csi_features(csi_data):
    """Analyze CSI data features including amplitude and phase."""
    amplitude = np.abs(csi_data)
    phase = np.angle(csi_data)
    
    # Basic statistics
    stats = {
        'amplitude_mean': np.mean(amplitude),
        'amplitude_std': np.std(amplitude),
        'amplitude_min': np.min(amplitude),
        'amplitude_max': np.max(amplitude),
        'phase_mean': np.mean(phase),
        'phase_std': np.std(phase),
        'phase_min': np.min(phase),
        'phase_max': np.max(phase)
    }
    
    return stats, amplitude, phase

def plot_csi_features(amplitude, phase, save_dir):
    """Create visualizations of CSI features."""
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot amplitude
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(amplitude[0], aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title('CSI Amplitude (First Antenna)')
    plt.xlabel('Sample Index')
    plt.ylabel('Subcarrier Index')
    
    # Plot phase
    plt.subplot(1, 2, 2)
    plt.imshow(phase[0], aspect='auto', cmap='viridis')
    plt.colorbar(label='Phase (radians)')
    plt.title('CSI Phase (First Antenna)')
    plt.xlabel('Sample Index')
    plt.ylabel('Subcarrier Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'csi_features.png'))
    plt.close()

def main():
    # Sample data paths
    real_file = 'CSI-dataset/Lab Dataset/coordinate 1-100/coordinate211.mat'
    imag_file = 'CSI-dataset/Lab Dataset/imaginary_part/imaginary211.mat'
    
    # Load and analyze sample
    print("Loading CSI data sample...")
    csi_data = load_csi_sample(real_file, imag_file)
    
    print("\nCSI Data Structure:")
    print(f"Shape: {csi_data.shape}")
    print(f"Number of antennas: {csi_data.shape[0]}")
    print(f"Number of subcarriers: {csi_data.shape[1]}")
    print(f"Number of samples: {csi_data.shape[2]}")
    
    print("\nAnalyzing CSI features...")
    stats, amplitude, phase = analyze_csi_features(csi_data)
    
    print("\nFeature Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nGenerating visualizations...")
    plot_csi_features(amplitude, phase, 'csi_analysis')
    print("Visualizations saved to 'csi_analysis/csi_features.png'")

if __name__ == "__main__":
    main()
