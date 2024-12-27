import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os

def load_cleaned_data():
    """Load the cleaned datasets."""
    features_train = pd.read_csv('cleaned_data/features_train.csv')
    features_val = pd.read_csv('cleaned_data/features_validation.csv')
    features_test = pd.read_csv('cleaned_data/features_test.csv')
    
    return features_train, features_val, features_test

def apply_pca(features_train, features_val, features_test, n_components=0.95):
    """Apply PCA to reduce dimensionality while preserving variance."""
    # Initialize imputer and PCA
    imputer = SimpleImputer(strategy='mean')
    pca = PCA(n_components=n_components)
    
    # First impute missing values
    print("Imputing missing values...")
    features_train_imputed = imputer.fit_transform(features_train)
    features_val_imputed = imputer.transform(features_val)
    features_test_imputed = imputer.transform(features_test)
    
    # Then apply PCA
    print("Applying PCA transformation...")
    features_train_pca = pca.fit_transform(features_train_imputed)
    features_val_pca = pca.transform(features_val_imputed)
    features_test_pca = pca.transform(features_test_imputed)
    
    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Create column names for PCA features
    pca_columns = [f'PC{i+1}' for i in range(features_train_pca.shape[1])]
    
    # Convert to DataFrames
    features_train_pca = pd.DataFrame(features_train_pca, columns=pca_columns)
    features_val_pca = pd.DataFrame(features_val_pca, columns=pca_columns)
    features_test_pca = pd.DataFrame(features_test_pca, columns=pca_columns)
    
    return features_train_pca, features_val_pca, features_test_pca, pca, cumulative_variance_ratio

def plot_explained_variance(cumulative_variance_ratio):
    """Plot cumulative explained variance ratio."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
            cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance vs. Number of PCA Components')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.close()

def main():
    # Create directory for reduced data
    os.makedirs('reduced_data', exist_ok=True)
    
    # Load cleaned data
    print("Loading cleaned data...")
    features_train, features_val, features_test = load_cleaned_data()
    
    # Apply PCA
    print("\nApplying PCA...")
    features_train_pca, features_val_pca, features_test_pca, pca, variance_ratio = apply_pca(
        features_train, features_val, features_test
    )
    
    # Plot explained variance
    print("Creating explained variance plot...")
    plot_explained_variance(variance_ratio)
    
    # Save reduced datasets
    print("\nSaving reduced datasets...")
    features_train_pca.to_csv('reduced_data/features_train_pca.csv', index=False)
    features_val_pca.to_csv('reduced_data/features_validation_pca.csv', index=False)
    features_test_pca.to_csv('reduced_data/features_test_pca.csv', index=False)
    
    # Print dimensionality reduction results
    print(f"\nOriginal number of features: {features_train.shape[1]}")
    print(f"Number of PCA components: {features_train_pca.shape[1]}")
    print(f"Explained variance with {features_train_pca.shape[1]} components: {variance_ratio[-1]:.4f}")
    
    # Print information about missing values
    print("\nMissing values summary:")
    print("Before imputation:")
    print(f"Training set NaN count: {features_train.isna().sum().sum()}")
    print(f"Validation set NaN count: {features_val.isna().sum().sum()}")
    print(f"Test set NaN count: {features_test.isna().sum().sum()}")

if __name__ == "__main__":
    main()
