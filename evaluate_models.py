import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

def load_models():
    """Load all trained models."""
    with open('models/zone_classifier.pkl', 'rb') as f:
        zone_classifier = pickle.load(f)
    with open('models/x_coordinate_regressor.pkl', 'rb') as f:
        x_regressor = pickle.load(f)
    with open('models/y_coordinate_classifier.pkl', 'rb') as f:
        y_classifier = pickle.load(f)
    with open('models/room_mapping.pkl', 'rb') as f:
        room_mapping = pickle.load(f)
    return zone_classifier, x_regressor, y_classifier, room_mapping

def load_test_data():
    """Load and validate test dataset."""
    X_test = pd.read_csv('reduced_data/features_test_pca.csv')
    y_test = pd.read_csv('cleaned_data/labels_test.csv')
    
    # Drop rows with NaN values
    valid_indices = ~y_test[['room', 'x', 'y']].isna().any(axis=1)
    X_test = X_test[valid_indices]
    y_test = y_test[valid_indices]
    
    print(f"\nTest data shape after removing NaN values:")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Number of samples removed: {(~valid_indices).sum()}")
    
    return X_test, y_test

def evaluate_zone_classifier(classifier, X_test, y_test, room_mapping):
    """Evaluate zone classifier performance."""
    # Convert room numbers to class indices
    reverse_mapping = {v: k for k, v in room_mapping.items()}
    y_true = y_test['room'].map(reverse_mapping)
    
    # Handle any rooms not in mapping
    missing_rooms = y_test[~y_test['room'].isin(reverse_mapping.keys())]['room'].unique()
    if len(missing_rooms) > 0:
        print(f"\nWarning: Found rooms in test set not present in training: {missing_rooms}")
        valid_indices = y_test['room'].isin(reverse_mapping.keys())
        X_test = X_test[valid_indices]
        y_true = y_true[valid_indices]
    
    # Get predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Zone Classification Confusion Matrix')
    plt.xlabel('Predicted Zone')
    plt.ylabel('True Zone')
    plt.savefig('evaluation/zone_confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nZone Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'confusion_matrix': conf_matrix,
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }

def evaluate_coordinate_prediction(x_regressor, y_classifier, X_test, y_test):
    """Evaluate coordinate prediction performance."""
    # X coordinate regression
    x_pred = x_regressor.predict(X_test)
    x_true = y_test['x']
    
    x_mse = mean_squared_error(x_true, x_pred)
    x_mae = mean_absolute_error(x_true, x_pred)
    x_r2 = r2_score(x_true, x_pred)
    
    print("\nX Coordinate Regression Metrics:")
    print(f"Mean Squared Error: {x_mse:.4f}")
    print(f"Mean Absolute Error: {x_mae:.4f}")
    print(f"R² Score: {x_r2:.4f}")
    
    # Y coordinate classification
    y_true_class = (y_test['y'] > 5.0).astype(int)
    y_pred_class = y_classifier.predict(X_test)
    
    print("\nY Coordinate Classification Report:")
    print(classification_report(y_true_class, y_pred_class))
    
    # Plot coordinate predictions
    plt.figure(figsize=(10, 6))
    y_pred_values = np.where(y_pred_class == 0, 4.480563296109769, 5.600704120137212)
    plt.scatter(x_true, y_test['y'], c='blue', label='Actual', alpha=0.6)
    plt.scatter(x_pred, y_pred_values, c='red', label='Predicted', alpha=0.6)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Coordinate Predictions vs Actual Values')
    plt.legend()
    plt.savefig('evaluation/coordinate_predictions.png')
    plt.close()
    
    return {
        'x_metrics': {
            'mse': x_mse,
            'mae': x_mae,
            'r2': x_r2
        },
        'y_classification': classification_report(y_true_class, y_pred_class, output_dict=True)
    }

def main():
    # Create evaluation directory
    import os
    os.makedirs('evaluation', exist_ok=True)
    
    # Load models and data
    print("Loading models and test data...")
    zone_classifier, x_regressor, y_classifier, room_mapping = load_models()
    X_test, y_test = load_test_data()
    
    # Evaluate zone classification
    print("\nEvaluating zone classifier...")
    zone_results = evaluate_zone_classifier(zone_classifier, X_test, y_test, room_mapping)
    
    # Evaluate coordinate prediction
    print("\nEvaluating coordinate prediction...")
    coord_results = evaluate_coordinate_prediction(x_regressor, y_classifier, X_test, y_test)
    
    # Save evaluation results
    print("\nSaving evaluation results...")
    results = {
        'zone_classification': zone_results,
        'coordinate_prediction': coord_results
    }
    with open('evaluation/results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nEvaluation complete! Results saved in 'evaluation' directory.")

if __name__ == "__main__":
    main()
