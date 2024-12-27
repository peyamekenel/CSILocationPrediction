import pandas as pd
import numpy as np

def analyze_features(file_path):
    print(f"\nAnalyzing {file_path}...")
    df = pd.read_csv(file_path)
    
    # Calculate mean and std for each feature type
    feature_types = ['csi_mean', 'csi_std', 'csi_max', 'rssi']
    
    for feature_type in feature_types:
        cols = [col for col in df.columns if col.startswith(feature_type)]
        if cols:
            values = df[cols].values.flatten()
            print(f"\n{feature_type} features statistics:")
            print(f"Mean: {np.mean(values):.6f}")
            print(f"Std: {np.std(values):.6f}")
            print(f"Min: {np.min(values):.6f}")
            print(f"Max: {np.max(values):.6f}")
    
    # Check for any missing values
    missing = df.isnull().sum().sum()
    print(f"\nTotal missing values: {missing}")

# Analyze all datasets
datasets = [
    'cleaned_data/features_train.csv',
    'cleaned_data/features_validation.csv',
    'cleaned_data/features_test.csv'
]

for dataset in datasets:
    analyze_features(dataset)
