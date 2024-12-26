import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configure matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def load_raw_csi_data():
    """Load raw CSI data preserving temporal structure."""
    print("Loading raw CSI data...")
    # Load amplitude and phase data
    amp_data = np.load('preprocessed/amplitude.npy')
    phase_data = np.load('preprocessed/phase.npy')
    
    # Load coordinates
    y = np.load('preprocessed/y.npy')
    
    return amp_data, phase_data, y

def preprocess_temporal_data(amp_data, phase_data):
    """Preprocess CSI data maintaining temporal structure with robust cleaning."""
    print("\nPreprocessing temporal data...")
    
    # Clean and validate input data
    def clean_data(data):
        # Replace inf/-inf with nan
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip extreme values to 3 standard deviations
        mean = np.nanmean(data)
        std = np.nanstd(data)
        data = np.clip(data, mean - 3*std, mean + 3*std)
        return data
    
    # Clean amplitude and phase data
    print("Cleaning amplitude data...")
    amp_data = clean_data(amp_data)
    print("Cleaning phase data...")
    phase_data = clean_data(phase_data)
    
    # Reshape data to (n_samples, timesteps, features)
    n_samples = amp_data.shape[0]
    n_timesteps = 100  # Use 100 timesteps (subsample from 1500)
    n_features = 90  # 3 antennas × 30 subcarriers
    
    # Subsample timesteps to reduce dimensionality
    step = 15  # 1500/100 = 15
    amp_temporal = amp_data[:, ::step, :]  # Shape: (n_samples, 100, 90)
    phase_temporal = phase_data[:, ::step, :]  # Shape: (n_samples, 100, 90)
    
    # Normalize each feature independently
    def normalize_features(data):
        # Reshape to 2D for normalization
        orig_shape = data.shape
        data_2d = data.reshape(-1, orig_shape[-1])
        
        # Replace any remaining inf values
        data_2d = np.nan_to_num(data_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize using robust scaling
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data_2d)
        
        # Reshape back to original shape
        return data_normalized.reshape(orig_shape), scaler
    
    print("Normalizing amplitude features...")
    amp_normalized, amp_scaler = normalize_features(amp_temporal)
    print("Normalizing phase features...")
    phase_normalized, phase_scaler = normalize_features(phase_temporal)
    
    # Combine normalized amplitude and phase
    X = np.concatenate([amp_normalized, phase_normalized], axis=2)
    
    print(f"Final data shape: {X.shape}")
    print("Data statistics after preprocessing:")
    print(f"Mean: {np.mean(X):.6f}")
    print(f"Std: {np.std(X):.6f}")
    print(f"Min: {np.min(X):.6f}")
    print(f"Max: {np.max(X):.6f}")
    
    return X, (amp_scaler, phase_scaler)

def create_lstm_model(input_shape):
    """Create LSTM model for coordinate prediction."""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='linear')  # Output layer for X,Y coordinates
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def evaluate_model(y_true, y_pred, model_name="LSTM"):
    """Evaluate model performance."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def main():
    """Main function to train and evaluate LSTM model."""
    # Create necessary directories
    os.makedirs('model_results', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    amp_data, phase_data, y = load_raw_csi_data()
    X, (amp_scaler, phase_scaler) = preprocess_temporal_data(amp_data, phase_data)
    
    # Create train-test split
    print("\nCreating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create and train LSTM model
    print("\nCreating LSTM model...")
    model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    model.summary()
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'model_results/lstm_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    results = evaluate_model(y_test, y_pred, "LSTM")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_results/lstm_training_history.png')
    plt.close()
    
    # Save predictions and test data
    np.save('model_results/lstm_predictions.npy', y_pred)
    np.save('model_results/lstm_test_actual.npy', y_test)
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv('model_results/lstm_results.csv', index=False)
    print("\nResults and predictions saved to 'model_results' directory")
    
    # Save scalers
    joblib.dump((amp_scaler, phase_scaler), 'model_results/lstm_scalers.joblib')
    print("Scalers saved to 'model_results/lstm_scalers.joblib'")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 5))
    
    # X coordinates
    plt.subplot(121)
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()],
             [y_test[:, 0].min(), y_test[:, 0].max()],
             'r--', lw=2)
    plt.xlabel('Actual X')
    plt.ylabel('Predicted X')
    plt.title('LSTM: X Coordinate Predictions')
    
    # Y coordinates
    plt.subplot(122)
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()],
             [y_test[:, 1].min(), y_test[:, 1].max()],
             'r--', lw=2)
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.title('LSTM: Y Coordinate Predictions')
    
    plt.tight_layout()
    plt.savefig('model_results/lstm_predictions.png')
    plt.close()

if __name__ == "__main__":
    main()
