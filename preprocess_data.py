import pandas as pd
import numpy as np
from scapy.all import rdpcap
import os
from typing import Tuple, List, Dict
import struct
from scipy import stats

class DataPreprocessor:
    def __init__(self, data_dir: str, locations_dir: str):
        self.data_dir = data_dir
        self.locations_dir = locations_dir
        # Load location information
        self.coords_df = pd.read_excel(os.path.join(locations_dir, 'location_coords.xlsx'))
        self.info_df = pd.read_excel(os.path.join(locations_dir, 'location_info.xlsx'))
        
    def extract_csi_from_packet(self, packet) -> np.ndarray:
        """Extract CSI data from packet payload."""
        if not packet.haslayer('Raw'):
            return None
            
        # Extract the raw payload
        payload = packet.getlayer('Raw').load
        
        try:
            # CSI data is in the payload after some header information
            # Convert bytes to complex numbers representing CSI
            # This is a simplified version - actual implementation would need to match
            # the specific format used in the Nexmon CSI extraction
            csi_data = []
            for i in range(0, len(payload), 4):
                if i + 4 <= len(payload):
                    # Convert 4 bytes into complex number (2 bytes real, 2 bytes imag)
                    real = struct.unpack('h', payload[i:i+2])[0]
                    imag = struct.unpack('h', payload[i+2:i+4])[0]
                    csi_data.append(complex(real, imag))
            
            return np.array(csi_data)
        except:
            return None
    
    def extract_features_from_pcap(self, pcap_file: str) -> Dict:
        """Extract CSI and RSSI features from a pcap file."""
        packets = rdpcap(pcap_file)
        
        # Extract CSI matrices from packets
        csi_matrices = []
        rssi_values = []
        
        for packet in packets:
            csi_data = self.extract_csi_from_packet(packet)
            if csi_data is not None:
                csi_matrices.append(csi_data)
                # Extract RSSI from RadioTap header if available
                if packet.haslayer('RadioTap'):
                    rssi = packet.getlayer('RadioTap').dBm_AntSignal
                    rssi_values.append(rssi)
                else:
                    rssi_values.append(None)  # Mark missing RSSI values
        
        if not csi_matrices:
            return None
            
        # Convert to numpy arrays
        csi_matrices = np.array(csi_matrices)
        rssi_values = np.array(rssi_values)
        
        # Filter out None values from RSSI
        rssi_values = np.array([x for x in rssi_values if x is not None])
        
        # Calculate amplitude and phase features from CSI
        csi_amplitude = np.abs(csi_matrices)
        csi_phase = np.angle(csi_matrices)
        
        # Calculate statistical features
        features = {
            # Amplitude features per subcarrier
            'csi_mean': np.mean(csi_amplitude, axis=0),
            'csi_std': np.std(csi_amplitude, axis=0),
            'csi_max': np.max(csi_amplitude, axis=0),
            'csi_min': np.min(csi_amplitude, axis=0),
            'csi_median': np.median(csi_amplitude, axis=0),
            'csi_q25': np.percentile(csi_amplitude, 25, axis=0),
            'csi_q75': np.percentile(csi_amplitude, 75, axis=0),
            'csi_skew': np.nan_to_num(stats.skew(csi_amplitude, axis=0)),
            'csi_kurtosis': np.nan_to_num(stats.kurtosis(csi_amplitude, axis=0)),
            
            # Phase features per subcarrier
            'csi_phase_mean': np.mean(csi_phase, axis=0),
            'csi_phase_std': np.std(csi_phase, axis=0),
            'csi_phase_max': np.max(csi_phase, axis=0),
            'csi_phase_min': np.min(csi_phase, axis=0),
            'csi_phase_median': np.median(csi_phase, axis=0),
            
            # Cross-subcarrier features
            'csi_corr_adjacent': np.array([np.corrcoef(csi_amplitude[:,i], csi_amplitude[:,i+1])[0,1] 
                                         if i < csi_amplitude.shape[1]-1 else 0 
                                         for i in range(csi_amplitude.shape[1])]),
            'csi_phase_diff': np.array([np.mean(np.diff(csi_phase[:,i:i+2], axis=1)) 
                                      if i < csi_phase.shape[1]-1 else 0 
                                      for i in range(csi_phase.shape[1])]),
            
            # RSSI features
            'rssi_mean': np.mean(rssi_values) if len(rssi_values) > 0 else 0,
            'rssi_std': np.std(rssi_values) if len(rssi_values) > 0 else 0,
            'rssi_max': np.max(rssi_values) if len(rssi_values) > 0 else 0,
            'rssi_min': np.min(rssi_values) if len(rssi_values) > 0 else 0,
            'rssi_median': np.median(rssi_values) if len(rssi_values) > 0 else 0
        }
        
        return features
    
    def get_location_labels(self, filename: str) -> Tuple[str, float, float]:
        """Extract location labels (room, x, y) from filename."""
        # Extract location ID from filename (e.g., "ref_204_1_3500.pcap" -> "204_1")
        location = '_'.join(filename.split('_')[1:3])
        
        # Get coordinates
        coords = self.coords_df[self.coords_df['location'] == location]
        if len(coords) == 0:
            return None, None, None
            
        room = location.split('_')[0]  # For classification
        x = coords.iloc[0]['x_coord']  # For regression
        y = coords.iloc[0]['y_coord']  # For regression
        
        return room, x, y
    
    def process_dataset(self, subset: str = 'ref') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process entire dataset and return features and labels."""
        features_list = []
        labels_list = []
        
        data_path = os.path.join(self.data_dir, subset)
        for filename in os.listdir(data_path):
            if not filename.endswith('.pcap'):
                continue
                
            print(f"Processing {filename}...")
            
            # Extract features
            features = self.extract_features_from_pcap(os.path.join(data_path, filename))
            if features is None:
                continue
                
            # Get labels
            room, x, y = self.get_location_labels(filename)
            if room is None:
                continue
                
            # Store results
            features_list.append(features)
            labels_list.append({
                'room': room,  # For classification
                'x': x,        # For regression
                'y': y         # For regression
            })
        
        # Convert to DataFrames
        features_df = pd.DataFrame(features_list)
        labels_df = pd.DataFrame(labels_list)
        
        return features_df, labels_df

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        data_dir='data',
        locations_dir='locations'
    )
    
    # Process training data
    print("Processing training data...")
    features_train, labels_train = preprocessor.process_dataset('ref')
    
    # Process test data
    print("Processing test data...")
    features_test, labels_test = preprocessor.process_dataset('test')
    
    # Save processed data
    features_train.to_csv('processed_data/features_train.csv', index=False)
    labels_train.to_csv('processed_data/labels_train.csv', index=False)
    features_test.to_csv('processed_data/features_test.csv', index=False)
    labels_test.to_csv('processed_data/labels_test.csv', index=False)
    
    print("Data preprocessing completed!")
    print(f"Training samples: {len(features_train)}")
    print(f"Test samples: {len(features_test)}")

if __name__ == "__main__":
    # Create processed_data directory if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)
    main()
