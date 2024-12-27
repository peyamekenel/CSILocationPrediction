import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data():
    """Load the preprocessed data."""
    features_train = pd.read_csv('processed_data/features_train.csv')
    labels_train = pd.read_csv('processed_data/labels_train.csv')
    features_test = pd.read_csv('processed_data/features_test.csv')
    labels_test = pd.read_csv('processed_data/labels_test.csv')
    
    # Convert string representations of arrays to actual arrays for CSI features
    def parse_array_string(s):
        if pd.isna(s):
            return np.array([])
        try:
            # Remove brackets and split by spaces
            s = s.strip('[]')
            # Split by spaces and filter out empty strings
            values = [float(x) for x in s.split() if x]
            return np.array(values)
        except (ValueError, AttributeError):
            print(f"Error parsing string: {s}")
            return np.array([])
    
    # List of all array-type features
    array_features = [
        'csi_mean', 'csi_std', 'csi_max', 'csi_min', 'csi_median',
        'csi_q25', 'csi_q75', 'csi_skew', 'csi_kurtosis',
        'csi_phase_mean', 'csi_phase_std', 'csi_phase_max',
        'csi_phase_min', 'csi_phase_median',
        'csi_corr_adjacent', 'csi_phase_diff'
    ]
    
    # Convert string representations of arrays to actual arrays for all array features
    for col in array_features:
        if col in features_train.columns:
            features_train[col] = features_train[col].apply(parse_array_string)
            features_test[col] = features_test[col].apply(parse_array_string)
    
    return features_train, labels_train, features_test, labels_test

def flatten_csi_features(features_train, features_test):
    """Flatten CSI array features into individual columns."""
    def flatten_single_df(df, name):
        flattened_features = []
        print(f"\nProcessing {name} dataset:")
        print(f"Input shape: {df.shape}")
        
        for idx, row in df.iterrows():
            try:
                flat_row = {}
                # Flatten CSI amplitude features
                for col in ['csi_mean', 'csi_std', 'csi_max', 'csi_min', 'csi_median', 'csi_q25', 'csi_q75', 'csi_skew', 'csi_kurtosis']:
                    if col in df.columns and isinstance(row[col], np.ndarray):
                        for i, val in enumerate(row[col]):
                            flat_row[f'{col}_{i}'] = float(val)
                
                # Flatten CSI phase features
                for col in ['csi_phase_mean', 'csi_phase_std', 'csi_phase_max', 'csi_phase_min', 'csi_phase_median']:
                    if col in df.columns and isinstance(row[col], np.ndarray):
                        for i, val in enumerate(row[col]):
                            flat_row[f'{col}_{i}'] = float(val)
                
                # Add cross-subcarrier features
                for col in ['csi_corr_adjacent', 'csi_phase_diff']:
                    if col in df.columns and isinstance(row[col], np.ndarray):
                        for i, val in enumerate(row[col]):
                            flat_row[f'{col}_{i}'] = float(val)
                
                # Add RSSI features
                for col in ['rssi_mean', 'rssi_std', 'rssi_max', 'rssi_min', 'rssi_median']:
                    if col in df.columns:
                        flat_row[col] = float(row[col])
                
                flattened_features.append(flat_row)
            except Exception as e:
                print(f"Error processing row {idx}:")
                print(f"Row data: {row}")
                print(f"Error details: {str(e)}")
                continue
        
        result_df = pd.DataFrame(flattened_features)
        print(f"Output shape: {result_df.shape}")
        return result_df
    
    train_flat = flatten_single_df(features_train, "training")
    test_flat = flatten_single_df(features_test, "test")
    
    # Ensure both dataframes have the same columns
    all_columns = sorted(list(set(train_flat.columns) | set(test_flat.columns)))
    for col in all_columns:
        if col not in train_flat.columns:
            train_flat[col] = 0
        if col not in test_flat.columns:
            test_flat[col] = 0
    
    train_flat = train_flat[all_columns]
    test_flat = test_flat[all_columns]
    
    return train_flat, test_flat

def normalize_features(features_train, features_test):
    """Standardize all features."""
    scaler = StandardScaler()
    features_train_norm = scaler.fit_transform(features_train)
    features_test_norm = scaler.transform(features_test)
    
    # Convert back to DataFrame with column names
    features_train_norm = pd.DataFrame(features_train_norm, columns=features_train.columns)
    features_test_norm = pd.DataFrame(features_test_norm, columns=features_test.columns)
    
    return features_train_norm, features_test_norm, scaler

def create_zone_labels(labels):
    """Create zone labels based on room numbers."""
    # Assuming rooms are numbered by floor/wing, group them into zones
    room_numbers = labels['room'].astype(int)
    # Create zones based on floor (first digit) and wing (second digit)
    zones = (room_numbers // 10).astype(int)
    return zones

def split_train_validation(features, labels, val_size=0.176):
    """Split training data into training and validation sets, using zones for stratification.
    Note: val_size=0.176 gives approximately 70-15-15 split since
    0.176 * 0.85 ≈ 0.15 (15% of total data)"""
    # Create zone labels for stratification
    zones = create_zone_labels(labels)
    
    print("\nZone distribution before splitting:")
    print(zones.value_counts())
    
    # Split while stratifying by zones
    features_train, features_val, labels_train, labels_val = train_test_split(
        features, labels, test_size=val_size, random_state=42,
        stratify=zones  # Stratify by zones instead of individual rooms
    )
    
    # Print distributions after splitting
    train_zones = create_zone_labels(labels_train)
    val_zones = create_zone_labels(labels_val)
    
    print("\nZone distribution in training set:")
    print(train_zones.value_counts())
    print("\nZone distribution in validation set:")
    print(val_zones.value_counts())
    
    print("\nRoom distribution in training set:")
    print(labels_train['room'].value_counts())
    print("\nRoom distribution in validation set:")
    print(labels_val['room'].value_counts())
    
    return features_train, features_val, labels_train, labels_val

def main():
    # Create directory for cleaned data
    os.makedirs('cleaned_data', exist_ok=True)
    
    # Load data
    print("Loading data...")
    features_train, labels_train, features_test, labels_test = load_data()
    
    # Print initial shapes
    print("\nInitial dataset shapes:")
    print(f"Features train: {features_train.shape}")
    print(f"Labels train: {labels_train.shape}")
    print(f"Features test: {features_test.shape}")
    print(f"Labels test: {labels_test.shape}")
    
    # Flatten CSI features
    print("\nFlattening CSI features...")
    features_train_flat, features_test_flat = flatten_csi_features(features_train, features_test)
    
    print("\nAfter flattening:")
    print(f"Features train: {features_train_flat.shape}")
    print(f"Features test: {features_test_flat.shape}")
    
    # Normalize features
    print("\nNormalizing features...")
    features_train_norm, features_test_norm, scaler = normalize_features(
        features_train_flat, features_test_flat
    )
    
    # Split training data into train and validation sets
    print("Splitting into train/validation sets...")
    (
        features_train_final,
        features_val,
        labels_train_final,
        labels_val,
    ) = split_train_validation(features_train_norm, labels_train)
    
    # Save processed datasets
    print("Saving processed datasets...")
    features_train_final.to_csv('cleaned_data/features_train.csv', index=False)
    features_val.to_csv('cleaned_data/features_validation.csv', index=False)
    features_test_norm.to_csv('cleaned_data/features_test.csv', index=False)
    
    labels_train_final.to_csv('cleaned_data/labels_train.csv', index=False)
    labels_val.to_csv('cleaned_data/labels_validation.csv', index=False)
    labels_test.to_csv('cleaned_data/labels_test.csv', index=False)
    
    # Print dataset sizes
    print("\nDataset sizes:")
    print(f"Training set: {len(features_train_final)} samples")
    print(f"Validation set: {len(features_val)} samples")
    print(f"Test set: {len(features_test_norm)} samples")


if __name__ == "__main__":
    main()
