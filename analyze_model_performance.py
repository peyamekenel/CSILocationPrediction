import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import joblib
import os

def load_data():
    """Load preprocessed data and model predictions."""
    X = np.load('preprocessed/X.npy')
    y = np.load('preprocessed/y.npy')
    return X, y

def load_models():
    """Load trained models."""
    rf_model = joblib.load('model_results/random_forest_model.joblib')
    gb_model = joblib.load('model_results/gradient_boosting_model.joblib')
    return rf_model, gb_model

def analyze_coordinate_distribution(y, title="Coordinate Distribution"):
    """Analyze and plot the distribution of X,Y coordinates."""
    plt.figure(figsize=(12, 5))
    
    # X coordinates
    plt.subplot(121)
    sns.histplot(y[:, 0], bins=30)
    plt.title('X Coordinate Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Count')
    
    # Y coordinates
    plt.subplot(122)
    sns.histplot(y[:, 1], bins=30)
    plt.title('Y Coordinate Distribution')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('model_results/coordinate_distribution.png')
    plt.close()

def plot_feature_importance(rf_model, X):
    """Plot feature importance from Random Forest model."""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 6))
    plt.title('Top 20 Feature Importances')
    plt.barh(range(20), importances[indices])
    plt.yticks(range(20), [f'Feature {i}' for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('model_results/feature_importance.png')
    plt.close()

def analyze_prediction_errors(y_true, y_pred, model_name):
    """Analyze prediction errors in detail."""
    errors = y_pred - y_true
    
    plt.figure(figsize=(12, 5))
    
    # X coordinate errors
    plt.subplot(121)
    sns.histplot(errors[:, 0], bins=30)
    plt.title(f'{model_name}: X Coordinate Error Distribution')
    plt.xlabel('Error (meters)')
    plt.ylabel('Count')
    
    # Y coordinate errors
    plt.subplot(122)
    sns.histplot(errors[:, 1], bins=30)
    plt.title(f'{model_name}: Y Coordinate Error Distribution')
    plt.xlabel('Error (meters)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'model_results/{model_name.lower().replace(" ", "_")}_error_distribution.png')
    plt.close()
    
    # Print error statistics
    print(f"\n{model_name} Error Statistics:")
    print(f"X Coordinate - Mean Error: {np.mean(errors[:, 0]):.2f}m, Std: {np.std(errors[:, 0]):.2f}m")
    print(f"Y Coordinate - Mean Error: {np.mean(errors[:, 1]):.2f}m, Std: {np.std(errors[:, 1]):.2f}m")

def plot_2d_predictions(y_true, y_pred, model_name):
    """Plot actual vs predicted locations in 2D space."""
    plt.figure(figsize=(10, 10))
    
    # Plot actual positions
    plt.scatter(y_true[:, 0], y_true[:, 1], c='blue', label='Actual', alpha=0.6)
    
    # Plot predicted positions
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', label='Predicted', alpha=0.6)
    
    # Draw lines between actual and predicted positions
    for i in range(len(y_true)):
        plt.plot([y_true[i, 0], y_pred[i, 0]], 
                [y_true[i, 1], y_pred[i, 1]], 
                'gray', alpha=0.2)
    
    plt.title(f'{model_name}: Actual vs Predicted Positions')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'model_results/{model_name.lower().replace(" ", "_")}_2d_predictions.png')
    plt.close()

def main():
    """Main function to analyze model performance."""
    print("Loading data and models...")
    X, y = load_data()
    rf_model, gb_model = load_models()
    
    # Analyze coordinate distribution
    print("\nAnalyzing coordinate distribution...")
    analyze_coordinate_distribution(y)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    plot_feature_importance(rf_model, X)
    
    # Make predictions
    print("\nGenerating predictions...")
    rf_pred = rf_model.predict(X)
    gb_pred = gb_model.predict(X)
    
    # Analyze prediction errors
    print("\nAnalyzing prediction errors...")
    analyze_prediction_errors(y, rf_pred, "Random Forest")
    analyze_prediction_errors(y, gb_pred, "Gradient Boosting")
    
    # Plot 2D predictions
    print("\nPlotting 2D predictions...")
    plot_2d_predictions(y, rf_pred, "Random Forest")
    plot_2d_predictions(y, gb_pred, "Gradient Boosting")
    
    print("\nAnalysis complete. Results saved in 'model_results' directory.")

if __name__ == "__main__":
    main()
