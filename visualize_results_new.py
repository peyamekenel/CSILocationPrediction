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
        'models/room_mapping.pkl'
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
    """Load all trained models and mappings using pickle."""
    models = {}
    try:
        # Load models with proper error handling
        with open('models/zone_classifier.pkl', 'rb') as f:
            models['zone'] = pickle.load(f)
        with open('models/x_coordinate_regressor.pkl', 'rb') as f:
            models['x_reg'] = pickle.load(f)
        with open('models/y_coordinate_classifier.pkl', 'rb') as f:
            models['y_class'] = pickle.load(f)
        with open('models/room_mapping.pkl', 'rb') as f:
            models['room_mapping'] = pickle.load(f)
        
        # Create inverse mapping for predictions
        models['class_to_room'] = {v: k for k, v in models['room_mapping'].items()}
        
        print("Successfully loaded models and mappings:")
        for name in models:
            print(f"- {name}")
        
        return models
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def plot_confusion_matrices(models, X_val, y_val):
    """Plot confusion matrices for room and y-coordinate classification."""
    # Create room classes
    # Get predictions first to ensure we have all possible rooms
    y_pred_room = models['zone'].predict(X_val)
    
    # Use the saved room mapping from training
    room_to_class = models['room_mapping']
    class_to_room = models['class_to_room']
    
    # Create a figure with subplots for room and y-coordinate classifiers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Convert validation room numbers to class indices using saved mapping
    y_val_room_class = y_val['room'].map(room_to_class)
    # Ensure predictions are integers
    y_pred_room_class = [int(cls) for cls in y_pred_room]
    
    # Get unique classes from both validation and predictions
    unique_classes = sorted(list(set(y_val_room_class) | set(y_pred_room_class)))
    # Convert class indices back to actual room numbers for labels
    room_labels = [class_to_room[cls] for cls in unique_classes]
    
    # Create confusion matrix with explicit labels
    cm_room = confusion_matrix(
        y_val_room_class, y_pred_room_class,
        labels=unique_classes
    )
    
    # Create heatmap with room number labels
    sns.heatmap(cm_room, annot=True, fmt='d', ax=ax1,
                xticklabels=room_labels,
                yticklabels=room_labels)
    
    ax1.set_title('Room Classification Confusion Matrix')
    ax1.set_xlabel('Predicted Room')
    ax1.set_ylabel('True Room')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Y-coordinate classifier confusion matrix (binary: 4.48 vs 5.60)
    y_pred_y = models['y_class'].predict(X_val)
    cm_y = confusion_matrix(y_val['y_class'], y_pred_y)
    sns.heatmap(cm_y, annot=True, fmt='d', ax=ax2)
    ax2.set_title('Y-Coordinate Classification Matrix')
    ax2.set_xlabel('Predicted Y-Class (0=4.48m, 1=5.60m)')
    ax2.set_ylabel('True Y-Class (0=4.48m, 1=5.60m)')
    
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices.png')
    plt.close()

def plot_coordinate_predictions(models, X_val, y_val):
    """Plot coordinate predictions with detailed visualization."""
    # Get predictions
    x_pred = models['x_reg'].predict(X_val)
    y_pred_class = models['y_class'].predict(X_val)
    
    # Convert y predictions to actual coordinates
    y_pred = np.where(y_pred_class == 0, 4.480563296109769, 5.600704120137212)
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(20, 7))
    
    # Plot 1: X coordinate regression performance
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(y_val['x'], x_pred, c='blue', alpha=0.6)
    min_x = min(y_val['x'].min(), x_pred.min())
    max_x = max(y_val['x'].max(), x_pred.max())
    ax1.plot([min_x, max_x], [min_x, max_x], 'r--', label='Perfect Prediction')
    ax1.set_title('X-Coordinate Regression Performance')
    ax1.set_xlabel('True X Coordinate (m)')
    ax1.set_ylabel('Predicted X Coordinate (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Y coordinate classification accuracy
    ax2 = plt.subplot(1, 3, 2)
    y_classes = ['4.48m', '5.60m']
    cm_y = confusion_matrix(y_val['y_class'], y_pred_class)
    sns.heatmap(cm_y, annot=True, fmt='d', ax=ax2, 
                xticklabels=y_classes, yticklabels=y_classes)
    ax2.set_title('Y-Coordinate Classification Performance')
    ax2.set_xlabel('Predicted Y-Coordinate')
    ax2.set_ylabel('True Y-Coordinate')
    
    # Plot 3: Floor plan visualization
    ax3 = plt.subplot(1, 3, 3)
    # Plot actual positions
    scatter_actual = ax3.scatter(y_val['x'], y_val['y'], 
                               c='blue', label='Actual', alpha=0.6)
    # Plot predicted positions
    scatter_pred = ax3.scatter(x_pred, y_pred, 
                             c='red', label='Predicted', alpha=0.6)
    
    # Draw arrows from actual to predicted positions
    for i in range(len(y_val)):
        ax3.arrow(y_val['x'].iloc[i], y_val['y'].iloc[i],
                 x_pred[i] - y_val['x'].iloc[i],
                 y_pred[i] - y_val['y'].iloc[i],
                 color='gray', alpha=0.3, head_width=0.2)
    
    # Add room annotations using consistent room numbers
    for room in y_val['room'].unique():
        room_data = y_val[y_val['room'] == room]
        center_x = room_data['x'].mean()
        center_y = room_data['y'].mean()
        # Use the actual room number (already correct in y_val['room'])
        ax3.annotate(f'Room {int(room)}', (center_x, center_y),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom')
    
    ax3.set_title('Indoor Location Predictions')
    ax3.set_xlabel('X Coordinate (m)')
    ax3.set_ylabel('Y Coordinate (m)')
    ax3.legend()
    ax3.grid(True)
    
    # Set proper y-axis limits for floor plan
    ax3.set_ylim(4.0, 6.0)
    
    plt.tight_layout()
    plt.savefig('visualizations/coordinate_predictions.png')
    plt.close()
    
    # Print performance metrics
    mse_x = mean_squared_error(y_val['x'], x_pred)
    r2_x = r2_score(y_val['x'], x_pred)
    y_accuracy = (y_pred_class == y_val['y_class']).mean()
    
    print("\nPerformance Metrics:")
    print(f"X-Coordinate Regression:")
    print(f"Mean Squared Error: {mse_x:.4f}")
    print(f"R² Score: {r2_x:.4f}")
    print(f"\nY-Coordinate Classification:")
    print(f"Accuracy: {y_accuracy:.4f}")

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
        
        print("\nVisualizations saved in 'visualizations' directory.")
    
    except Exception as e:
        print(f"Error occurred while generating visualizations: {str(e)}")
        raise

if __name__ == '__main__':
    main()
