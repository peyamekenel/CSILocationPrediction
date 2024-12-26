import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
import os

def load_data():
    """Load preprocessed data and model predictions."""
    X = np.load('preprocessed/X.npy')
    y = np.load('preprocessed/y.npy')
    return X, y

def analyze_coordinate_distribution(y):
    """Analyze and visualize the distribution of coordinates."""
    plt.figure(figsize=(15, 5))
    
    # Plot X coordinate distribution
    plt.subplot(121)
    plt.hist(y[:, 0], bins=30, alpha=0.7)
    plt.title('X Coordinate Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Frequency')
    
    # Plot Y coordinate distribution
    plt.subplot(122)
    plt.hist(y[:, 1], bins=30, alpha=0.7)
    plt.title('Y Coordinate Distribution')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('model_results/coordinate_distribution.png')
    plt.close()
    
    # Print coordinate statistics
    print("\nCoordinate Statistics:")
    print(f"X range: {y[:, 0].min():.2f} to {y[:, 0].max():.2f}")
    print(f"Y range: {y[:, 1].min():.2f} to {y[:, 1].max():.2f}")
    print(f"X mean: {y[:, 0].mean():.2f}, std: {y[:, 0].std():.2f}")
    print(f"Y mean: {y[:, 1].mean():.2f}, std: {y[:, 1].std():.2f}")

def analyze_feature_importance():
    """Analyze feature importance from Random Forest model."""
    try:
        rf_model = joblib.load('model_results/random_forest_model.joblib')
        importances = rf_model.feature_importances_
        
        # Create feature names for all 1080 features
        feature_types = ['amp_mean', 'amp_std', 'amp_max', 'amp_min', 'amp_median', 'amp_q25', 'amp_q75',
                        'phase_mean', 'phase_std', 'phase_max', 'phase_min', 'phase_median', 'phase_q25', 'phase_q75']
        feature_names = []
        for ant in range(3):
            for subcarrier in range(30):  # 30 subcarriers per antenna
                for feat in feature_types:
                    feature_names.append(f"ant{ant+1}_sub{subcarrier+1}_{feat}")
        
        # Plot top 20 most important features
        plt.figure(figsize=(12, 6))
        indices = np.argsort(importances)[-20:]
        plt.barh(range(20), importances[indices])
        plt.yticks(range(20), [feature_names[i] for i in indices], fontsize=8)
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('model_results/feature_importance.png')
        plt.close()
        
        print("\nTop 10 Most Important Features:")
        for i in indices[-10:]:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
            
    except Exception as e:
        print(f"Error analyzing feature importance: {str(e)}")
        
def create_pca_model(X, n_components=None):
    """Create and fit PCA model."""
    if n_components is None:
        pca = PCA()
    else:
        pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def analyze_pca_components(X):
    """Analyze principal components of the feature space."""
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
            cumulative_variance_ratio, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_results/pca_analysis.png')
    plt.close()
    
    # Print PCA statistics
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"\nPCA Analysis:")
    print(f"Number of components explaining 95% of variance: {n_components_95}")
    print(f"Explained variance by first 10 components: {explained_variance_ratio[:10].sum():.4f}")

def main():
    """Main analysis function."""
    print("Loading data...")
    X, y = load_data()
    
    print("\nAnalyzing coordinate distribution...")
    analyze_coordinate_distribution(y)
    
    print("\nAnalyzing feature importance...")
    analyze_feature_importance()
    
    
    print("\nAnalyzing feature space using PCA...")
    analyze_pca_components(X)

if __name__ == "__main__":
    main()
