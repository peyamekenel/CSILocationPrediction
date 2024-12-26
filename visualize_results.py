import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

def load_results():
    """Load results from all models."""
    # Load traditional model results
    model_results = pd.read_csv('model_results/model_comparison.csv')
    
    # Load LSTM results
    lstm_results = pd.read_csv('model_results/lstm_results.csv')
    
    return model_results, lstm_results

def load_test_data():
    """Load the original test data and predictions."""
    # Load coordinate scaler
    coord_scaler = joblib.load('preprocessed/coord_scaler.joblib')
    
    # Load actual coordinates
    y = np.load('preprocessed/y.npy')
    
    # Split into train/test (using same split as in training)
    from sklearn.model_selection import train_test_split
    _, y_test = train_test_split(y, test_size=0.2, random_state=42)
    
    return y_test, coord_scaler

def create_prediction_plots(y_test, coord_scaler, model_results, lstm_results):
    """Create plots comparing actual vs predicted coordinates."""
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Actual vs Predicted Indoor Locations\nby Different Models', fontsize=16, y=0.95)
    
    # Get actual coordinates
    actual_coords = coord_scaler.inverse_transform(y_test)
    
    # Plot for each model
    models = ['Random Forest', 'Gradient Boosting', 'Neural Network', 'LSTM']
    predictions = []
    
    # Load predictions for traditional models
    for model in models[:-1]:
        pred_file = f'model_results/{model.lower().replace(" ", "_")}_predictions.npy'
        pred = np.load(pred_file)
        predictions.append(coord_scaler.inverse_transform(pred))
    
    # Load LSTM predictions
    lstm_pred = np.load('model_results/lstm_predictions.npy')
    predictions.append(coord_scaler.inverse_transform(lstm_pred))
    
    # Create scatter plots
    for idx, (model, pred) in enumerate(zip(models, predictions)):
        ax = axes[idx // 2, idx % 2]
        
        # Plot actual vs predicted points
        scatter = ax.scatter(actual_coords[:, 0], actual_coords[:, 1], 
                           c='blue', label='Actual', alpha=0.5)
        ax.scatter(pred[:, 0], pred[:, 1], 
                  c='red', label='Predicted', alpha=0.5)
        
        # Draw lines between actual and predicted points
        for actual, predicted in zip(actual_coords, pred):
            ax.plot([actual[0], predicted[0]], [actual[1], predicted[1]], 
                   'g-', alpha=0.1)
        
        # Calculate average error
        errors = np.sqrt(np.sum((actual_coords - pred) ** 2, axis=1))
        avg_error = np.mean(errors)
        
        ax.set_title(f'{model}\nAverage Error: {avg_error:.2f}m')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.legend()
        ax.grid(True)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('model_results/prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_error_distribution(y_test, coord_scaler, model_results, lstm_results):
    """Create error distribution plots for all models."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Error Distribution Analysis', fontsize=16, y=0.95)
    
    # Prepare data
    models = ['Random Forest', 'Gradient Boosting', 'Neural Network', 'LSTM']
    errors = []
    
    actual_coords = coord_scaler.inverse_transform(y_test)
    
    # Calculate errors for each model
    for model in models[:-1]:
        pred_file = f'model_results/{model.lower().replace(" ", "_")}_predictions.npy'
        pred = np.load(pred_file)
        pred = coord_scaler.inverse_transform(pred)
        error = np.sqrt(np.sum((actual_coords - pred) ** 2, axis=1))
        errors.append(error)
    
    
    # Add LSTM errors
    lstm_pred = np.load('model_results/lstm_predictions.npy')
    lstm_pred = coord_scaler.inverse_transform(lstm_pred)
    lstm_error = np.sqrt(np.sum((actual_coords - lstm_pred) ** 2, axis=1))
    errors.append(lstm_error)
    
    # Create box plot
    ax1.boxplot(errors, labels=models)
    ax1.set_title('Error Distribution Across Models')
    ax1.set_ylabel('Error (meters)')
    ax1.grid(True)
    
    # Create error histogram
    for error, model in zip(errors, models):
        ax2.hist(error, bins=20, alpha=0.5, label=model)
    
    ax2.set_title('Error Distribution Histogram')
    ax2.set_xlabel('Error (meters)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_results/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to create visualization plots."""
    try:
        # Load results
        model_results, lstm_results = load_results()
        
        # Load test data
        y_test, coord_scaler = load_test_data()
        
        # Create visualization plots
        create_prediction_plots(y_test, coord_scaler, model_results, lstm_results)
        create_error_distribution(y_test, coord_scaler, model_results, lstm_results)
        
        print("Visualization plots have been created successfully!")
        print("Check 'model_results/prediction_comparison.png' and 'model_results/error_distribution.png'")
        
    except Exception as e:
        print(f"Error creating visualization plots: {str(e)}")

if __name__ == "__main__":
    main()
