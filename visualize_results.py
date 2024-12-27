import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import pickle
import os

# Set style for better looking plots
plt.style.use('default')
sns.set_style("whitegrid")

def check_files_exist():
    """Check if all required files exist."""
    required_files = [
        'reduced_data/features_validation_pca.csv',
        'cleaned_data/labels_validation.csv',
        'models/zone_classifier.pkl',
        'models/x_coordinate_regressor.pkl',
        'models/y_coordinate_classifier.pkl',
        'models/coordinate_regressor.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"- {file}")
        return False
    return True

def load_data():
    """Load validation data and ensure correct column names."""
    # Load the validation data
    X_val = pd.read_csv('reduced_data/features_validation_pca.csv')
    y_val = pd.read_csv('cleaned_data/labels_validation.csv')
    
    # Ensure y_val has the expected columns
    if 'y_class' not in y_val.columns:
        y_val['y_class'] = (y_val['y'] > 5.0).astype(int)
    
    return X_val, y_val

def load_models():
    """Load all trained models using pickle."""
    models = {}
    try:
        # Load models with proper error handling
        with open('models/zone_classifier.pkl', 'rb') as f:
            models['zone'] = pickle.load(f)
        with open('models/x_coordinate_regressor.pkl', 'rb') as f:
            models['x_reg'] = pickle.load(f)
        with open('models/y_coordinate_classifier.pkl', 'rb') as f:
            models['y_class'] = pickle.load(f)
        
        print("Successfully loaded models:")
        for name in models:
            print(f"- {name}")
        
        return models
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def plot_confusion_matrices(models, X_val, y_val):
    # Create a figure with subplots for zone and y-coordinate classifiers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Zone classifier confusion matrix
    y_pred_zone = models['zone'].predict(X_val)
    cm_zone = confusion_matrix(y_val['room_class'], y_pred_zone)
    sns.heatmap(cm_zone, annot=True, fmt='d', ax=ax1)
    ax1.set_title('Zone Classification Confusion Matrix')
    ax1.set_xlabel('Predicted Zone')
    ax1.set_ylabel('True Zone')
    
    # Y-coordinate classifier confusion matrix
    y_pred_y = models['y_class'].predict(X_val)
    cm_y = confusion_matrix(y_val['y_class'], y_pred_y)
    sns.heatmap(cm_y, annot=True, fmt='d', ax=ax2)
    ax2.set_title('Y-Coordinate Classification Confusion Matrix')
    ax2.set_xlabel('Predicted Class')
    ax2.set_ylabel('True Class')
    
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices.png')
    plt.close()

def plot_coordinate_predictions(models, X_val, y_val):
    # Get predictions
    x_pred = models['x_reg'].predict(X_val)
    y_pred = models['y_class'].predict(X_val)
    
    # Create figure with three subplots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: X coordinate predictions
    plt.subplot(1, 3, 1)
    plt.scatter(range(len(y_val)), y_val['x'], c='blue', label='Actual', alpha=0.6)
    plt.scatter(range(len(y_val)), x_pred, c='red', label='Predicted', alpha=0.6)
    plt.title('X Coordinate Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('X Coordinate')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Y coordinate classification accuracy
    plt.subplot(1, 3, 2)
    correct = (y_pred == y_val['y_class']).astype(int)
    plt.bar(range(len(y_val)), correct, alpha=0.6)
    plt.title('Y Coordinate Classification Accuracy')
    plt.xlabel('Sample Index')
    plt.ylabel('Correct (1) / Incorrect (0)')
    plt.grid(True)
    
    # Plot 3: Floor plan visualization
    plt.subplot(1, 3, 3)
    # Plot actual positions
    plt.scatter(y_val['x'], y_val['y'], c='blue', label='Actual', alpha=0.6)
    # Plot predicted positions (using binary y_class predictions)
    y_pred_coord = np.where(y_pred == 1, 1.0, 0.0)  # Convert to actual y-coordinates
    plt.scatter(x_pred, y_pred_coord, c='red', label='Predicted', alpha=0.6)
    
    # Draw arrows from actual to predicted positions
    for i in range(len(y_val)):
        plt.arrow(y_val['x'].iloc[i], y_val['y'].iloc[i],
                 x_pred[i] - y_val['x'].iloc[i],
                 y_pred_coord[i] - y_val['y'].iloc[i],
                 color='gray', alpha=0.3, head_width=0.1)
    
    plt.title('Indoor Location Predictions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/coordinate_predictions.png')
    plt.close()

def main():
    # Check if required files exist
    if not check_files_exist():
        return
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    try:
        # Load data and models
        print("Loading data and models...")
        X_val, y_val = load_data()
        models = load_models()
        
        # Create visualizations
        print("Creating confusion matrices...")
        plot_confusion_matrices(models, X_val, y_val)
        
        print("Creating coordinate prediction plots...")
        plot_coordinate_predictions(models, X_val, y_val)
        
        # Calculate and print performance metrics
        x_pred = models['x_reg'].predict(X_val)
        y_pred = models['y_class'].predict(X_val)
        
        print("\nModel Performance Metrics:")
        print("X Coordinate Regression:")
        print(f"Mean Squared Error: {mean_squared_error(y_val['x'], x_pred):.4f}")
        print(f"R² Score: {r2_score(y_val['x'], x_pred):.4f}")
        print("\nY Coordinate Classification:")
        correct = (y_pred == y_val['y_class']).mean()
        print(f"Accuracy: {correct:.4f}")
        
        print("\nVisualizations saved in 'visualizations' directory.")
    
    except Exception as e:
        print(f"Error occurred while generating visualizations: {str(e)}")
        raise

if __name__ == '__main__':
    main()
