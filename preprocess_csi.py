import os
import numpy as np
import scipy.io
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

def load_csi_data(real_path, imag_path):
    """Load and combine real and imaginary parts of CSI data."""
    real_data = scipy.io.loadmat(real_path)['myData']
    imag_data = scipy.io.loadmat(imag_path)['myData']
    return real_data + 1j * imag_data

def extract_coordinates(filename):
    """Extract X,Y coordinates from filename."""
    if 'coordinate' in filename:
        coord_str = filename.split('coordinate')[1].split('.')[0]
    else:
        coord_str = filename.split('imaginary')[1].split('.')[0]
    return int(coord_str[0]), int(coord_str[1:])

def phase_correction(phase_data):
    """Perform phase correction by unwrapping across subcarriers."""
    return np.unwrap(phase_data, axis=1)

def clip_and_clean(data, lower_percentile=1, upper_percentile=99):
    """Clip extreme values and replace infinities/NaNs with median."""
    data = np.array(data)
    # Replace inf/-inf with NaN
    data[~np.isfinite(data)] = np.nan
    
    # Get percentile values for valid data
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return np.zeros_like(data)
    
    lower = np.percentile(valid_data, lower_percentile)
    upper = np.percentile(valid_data, upper_percentile)
    
    # Clip the data
    data = np.clip(data, lower, upper)
    
    # Replace remaining NaNs with median
    median_val = np.nanmedian(data)
    data[np.isnan(data)] = median_val
    
    return data

def extract_features(csi_data):
    """Extract features from CSI data with robust handling of extreme values."""
    # Get amplitude and corrected phase
    amplitude = np.abs(csi_data)
    phase = phase_correction(np.angle(csi_data))
    
    # Initialize feature array
    n_antennas, n_subcarriers, n_samples = csi_data.shape
    features = []
    
    # Per antenna features
    for ant in range(n_antennas):
        # Clean amplitude data
        amp_data = amplitude[ant]
        amp_data = clip_and_clean(amp_data)
        
        # Clean phase data
        phase_data = phase[ant]
        phase_data = clip_and_clean(phase_data, lower_percentile=5, upper_percentile=95)
        
        # Amplitude features (across samples)
        amp_mean = np.mean(amp_data, axis=1)
        amp_std = np.std(amp_data, axis=1)
        amp_max = np.max(amp_data, axis=1)
        amp_min = np.min(amp_data, axis=1)
        amp_median = np.median(amp_data, axis=1)
        
        # Phase features (across samples)
        phase_mean = np.mean(phase_data, axis=1)
        phase_std = np.std(phase_data, axis=1)
        phase_median = np.median(phase_data, axis=1)
        
        # Additional robust features
        amp_q25 = np.percentile(amp_data, 25, axis=1)
        amp_q75 = np.percentile(amp_data, 75, axis=1)
        phase_q25 = np.percentile(phase_data, 25, axis=1)
        phase_q75 = np.percentile(phase_data, 75, axis=1)
        
        # Combine features
        ant_features = np.concatenate([
            amp_mean, amp_std, amp_max, amp_min, amp_median,
            amp_q25, amp_q75,
            phase_mean, phase_std, phase_median,
            phase_q25, phase_q75
        ])
        features.extend(ant_features.tolist())
    
    return np.array(features)

def save_temporal_data(csi_data_list):
    """Save raw CSI data for temporal analysis."""
    n_samples = len(csi_data_list)
    n_timesteps = 1500  # Original number of samples
    n_features = 90  # 3 antennas × 30 subcarriers
    
    # Initialize arrays
    amplitude = np.zeros((n_samples, n_timesteps, n_features))
    phase = np.zeros((n_samples, n_timesteps, n_features))
    
    # Extract amplitude and phase
    for i, csi_data in enumerate(csi_data_list):
        for ant in range(3):
            for sub in range(30):
                idx = ant * 30 + sub
                amplitude[i, :, idx] = np.abs(csi_data[ant][sub])
                phase[i, :, idx] = np.angle(csi_data[ant][sub])
    
    # Save temporal features
    print("\nSaving temporal data for LSTM...")
    np.save('preprocessed/amplitude.npy', amplitude)
    np.save('preprocessed/phase.npy', phase)
    print(f"Saved temporal features:")
    print(f"Amplitude shape: {amplitude.shape}")
    print(f"Phase shape: {phase.shape}")

def process_dataset(base_dir):
    """
    Veri setini işleyerek eğitim için hazırlar.
    
    Bu fonksiyon şu adımları gerçekleştirir:
    1. CSI verilerini yükler
    2. Özellik çıkarımı yapar
    3. Koordinat verilerini normalleştirir
    4. İşlenmiş verileri kaydeder
    
    Args:
        base_dir (str): Ham CSI verilerinin bulunduğu dizin
        
    Returns:
        None: İşlenmiş veriler 'preprocessed/' dizinine kaydedilir
        
    Raises:
        FileNotFoundError: Veri dizini bulunamazsa
        ValueError: Veri formatı geçersizse
    """
    features_list = []  # Statistical Features
    coordinates_list = []  # Coordinates
    csi_data_list = []  # Raw CSI data for temporal analysis
    
    # Process Lab Dataset
    lab_dir = os.path.join(base_dir, 'Lab Dataset')
    coord_dirs = [d for d in os.listdir(lab_dir) if d.startswith('coordinate') and d != 'imaginary_part']
    
    print(f"Found {len(coord_dirs)} coordinate directories")
    
    # Create output directory
    os.makedirs('preprocessed', exist_ok=True)
    
    for coord_dir in coord_dirs:
        coord_path = os.path.join(lab_dir, coord_dir)
        for file in os.listdir(coord_path):
            if file.endswith('.mat'):
                # Get paths for real and imaginary parts
                real_path = os.path.join(coord_path, file)
                imag_file = f"imaginary{file.split('coordinate')[1]}"
                imag_path = os.path.join(lab_dir, 'imaginary_part', imag_file)
                
                if os.path.exists(imag_path):
                    try:
                        # Load and process CSI data
                        csi_data = load_csi_data(real_path, imag_path)
                        csi_data_list.append(csi_data)  # Save raw CSI data for LSTM
                        
                        # Extract statistical features
                        features = extract_features(csi_data)
                        features_list.append(features)
                        
                        # Extract coordinates
                        x_coord, y_coord = extract_coordinates(file)
                        coordinates_list.append([x_coord, y_coord])
                        
                        if len(features_list) % 50 == 0:
                            print(f"Processed {len(features_list)} samples...")
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
                        continue
    
    print(f"\nTotal samples processed: {len(features_list)}")
    
    X = np.array(features_list)
    coordinates = np.array(coordinates_list)
    
    # Print coordinate statistics before normalization
    print("\nCoordinate Statistics before normalization:")
    print(f"X range: {coordinates[:, 0].min():.2f} to {coordinates[:, 0].max():.2f}")
    print(f"Y range: {coordinates[:, 1].min():.2f} to {coordinates[:, 1].max():.2f}")
    
    # Normalize coordinates to reasonable indoor positioning range
    if np.max(np.abs(coordinates[:, 1])) > 100:  # If Y-coordinates are unreasonably large
        print("\nWARNING: Y-coordinates appear to be incorrectly scaled!")
        print("Applying scaling correction to Y-coordinates...")
        coordinates[:, 1] = coordinates[:, 1] / 100.0  # Scale down Y-coordinates
    
    # Ensure all coordinates are positive
    coordinates[:, 0] = coordinates[:, 0] - coordinates[:, 0].min()
    coordinates[:, 1] = coordinates[:, 1] - coordinates[:, 1].min()
    
    print("\nCoordinate Statistics after normalization:")
    print(f"X range: {coordinates[:, 0].min():.2f} to {coordinates[:, 0].max():.2f}")
    print(f"Y range: {coordinates[:, 1].min():.2f} to {coordinates[:, 1].max():.2f}")
    
    y = coordinates
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Normalize and save coordinates
    coord_scaler = StandardScaler()
    y_scaled = coord_scaler.fit_transform(y)
    
    # Save preprocessed data and scalers
    try:
        print("\nSaving preprocessed data...")
        np.save('preprocessed/X.npy', X_scaled)
        np.save('preprocessed/y.npy', y_scaled)
        joblib.dump(scaler, 'preprocessed/scaler.joblib')
        joblib.dump(coord_scaler, 'preprocessed/coord_scaler.joblib')
        print("Data successfully saved to 'preprocessed' directory")
        
        # Save feature information
        n_features = X.shape[1]
        feature_info = {
            'n_features': n_features,
            'n_samples': len(X),
            'feature_names': [
                f'antenna{i+1}_{feat_type}_{stat_type}'
                for i in range(3)  # 3 antennas
                for feat_type in ['amp', 'phase']
                for stat_type in ['mean', 'std', 'max', 'min', 'median', 'q25', 'q75']
            ]
        }
        
        with open('preprocessed/feature_info.txt', 'w') as f:
            f.write("CSI Dataset Feature Information\n")
            f.write("============================\n\n")
            for key, value in feature_info.items():
                if key == 'feature_names':
                    f.write(f"{key}:\n")
                    for i, name in enumerate(value):
                        f.write(f"  {i}: {name}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            # Add statistics
            f.write("\nFeature Statistics (before scaling):\n")
            f.write("--------------------------------\n")
            f.write(f"Mean: {np.mean(X):.6f}\n")
            f.write(f"Std: {np.std(X):.6f}\n")
            f.write(f"Min: {np.min(X):.6f}\n")
            f.write(f"Max: {np.max(X):.6f}\n")
    except Exception as e:
        print(f"\nError saving preprocessed data: {str(e)}")
        raise
    
    # Save temporal data for LSTM
    save_temporal_data(csi_data_list)
    
    return X_scaled, y, scaler

def main():
    print("Starting data preprocessing...")
    try:
        X, y, scaler = process_dataset('CSI-dataset')
        
        print("\nPreprocessing completed successfully!")
        print(f"Statistical Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print("\nFeature statistics after normalization:")
        print(f"Mean: {X.mean():.6f}")
        print(f"Std: {X.std():.6f}")
        print(f"Min: {X.min():.6f}")
        print(f"Max: {X.max():.6f}")
        print("\nPreprocessed data and documentation saved to 'preprocessed' directory")
        
    except Exception as e:
        print(f"\nError in preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
