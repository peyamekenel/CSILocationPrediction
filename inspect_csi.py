import scipy.io
import numpy as np
import os
from pathlib import Path

def analyze_coordinate_from_filename(filename):
    # Extract numbers from filename like 'coordinate211.mat' or 'imaginary211.mat' -> (2, 11)
    if 'coordinate' in filename:
        coord_str = filename.split('coordinate')[1].split('.')[0]
    else:
        coord_str = filename.split('imaginary')[1].split('.')[0]
    x = int(coord_str[0])
    y = int(coord_str[1:])
    return x, y

def analyze_mat_file(filepath):
    print(f"\nAnalyzing file: {filepath}")
    data = scipy.io.loadmat(filepath)
    
    # Analyze myData structure
    csi_data = data['myData']
    print(f"CSI Data Shape: {csi_data.shape}")
    print(f"Number of dimensions: {csi_data.ndim}")
    print(f"Data type: {csi_data.dtype}")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(f"Mean value: {np.mean(csi_data)}")
    print(f"Standard deviation: {np.std(csi_data)}")
    print(f"Min value: {np.min(csi_data)}")
    print(f"Max value: {np.max(csi_data)}")
    
    # Check if data might contain complex values
    if np.iscomplexobj(csi_data):
        print("\nComplex data detected!")
        print("Contains both amplitude and phase information")
        print(f"Sample amplitude: {np.abs(csi_data[0,0,:5])}")
        print(f"Sample phase: {np.angle(csi_data[0,0,:5])}")
    else:
        print("\nReal-valued data detected")
        print("This might be amplitude data")
        print(f"Sample values: {csi_data[0,0,:5]}")
    
    # Extract coordinates from filename
    filename = os.path.basename(filepath)
    x, y = analyze_coordinate_from_filename(filename)
    print(f"\nExtracted coordinates: x={x}, y={y}")

# Analyze sample files from both real and imaginary parts
print("=== Analyzing Lab Dataset ===")
real_file = 'CSI-dataset/Lab Dataset/coordinate 1-100/coordinate211.mat'
imag_file = 'CSI-dataset/Lab Dataset/imaginary_part/imaginary211.mat'

def combine_csi_data(real_path, imag_path):
    real_data = scipy.io.loadmat(real_path)['myData']
    imag_data = scipy.io.loadmat(imag_path)['myData']
    
    # Combine into complex numbers
    complex_data = real_data + 1j * imag_data
    
    print("\nCombined CSI Data Analysis:")
    print(f"Shape: {complex_data.shape}")
    print("\nCSI Components:")
    print(f"Amplitude (first 5 samples): {np.abs(complex_data[0,0,:5])}")
    print(f"Phase (first 5 samples): {np.angle(complex_data[0,0,:5])}")
    
    # Analyze dimensions
    print("\nDimension Analysis:")
    print(f"Number of transmitters/receivers: {complex_data.shape[0]}")
    print(f"Number of subcarriers: {complex_data.shape[1]}")
    print(f"Number of samples: {complex_data.shape[2]}")

print("\nAnalyzing Real Part:")
analyze_mat_file(real_file)

print("\nAnalyzing Imaginary Part:")
analyze_mat_file(imag_file)

print("\nAnalyzing Combined Complex CSI Data:")
combine_csi_data(real_file, imag_file)
