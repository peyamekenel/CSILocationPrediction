import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
import os

def load_preprocessed_data():
    """Load preprocessed CSI data and coordinates."""
    X = np.load('preprocessed/X.npy')
    y = np.load('preprocessed/y.npy')
    return X, y

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """Create train-test split with stratification."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print regression metrics."""
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

def plot_predictions(y_true, y_pred, model_name, save_dir='model_results'):
    """Plot actual vs predicted coordinates."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # X coordinates
    plt.subplot(121)
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_true[:, 0].min(), y_true[:, 0].max()], 
             [y_true[:, 0].min(), y_true[:, 0].max()], 
             'r--', lw=2)
    plt.xlabel('Actual X')
    plt.ylabel('Predicted X')
    plt.title(f'{model_name}: X Coordinate Predictions')
    
    # Y coordinates
    plt.subplot(122)
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_true[:, 1].min(), y_true[:, 1].max()],
             [y_true[:, 1].min(), y_true[:, 1].max()],
             'r--', lw=2)
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.title(f'{model_name}: Y Coordinate Predictions')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name.lower().replace(" ", "_")}_predictions.png')
    plt.close()

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest model."""
    print("\nTraining Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=1000,  # Increased number of trees
        max_depth=20,       # Reduced to prevent overfitting
        min_samples_split=5,  # Increased for better generalization
        min_samples_leaf=4,   # Increased for robustness
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,     # Enable out-of-bag score estimation
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Random Forest")
    plot_predictions(y_test, y_pred, "Random Forest")
    
    # Save model
    joblib.dump(rf_model, 'model_results/random_forest_model.joblib')
    return metrics

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train and evaluate Gradient Boosting model."""
    print("\nTraining Gradient Boosting Regressor...")
    base_gb = GradientBoostingRegressor(
        n_estimators=300,     # Increased number of estimators
        learning_rate=0.01,   # Reduced for better convergence
        max_depth=6,          # Reduced to prevent overfitting
        min_samples_split=5,  # Increased for better generalization
        min_samples_leaf=4,   # Increased for robustness
        subsample=0.8,
        validation_fraction=0.2,  # Added validation for early stopping
        n_iter_no_change=10,     # Early stopping patience
        random_state=42
    )
    gb_model = MultiOutputRegressor(base_gb, n_jobs=-1)
    gb_model.fit(X_train, y_train)
    
    y_pred = gb_model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Gradient Boosting")
    plot_predictions(y_test, y_pred, "Gradient Boosting")
    
    # Save model
    joblib.dump(gb_model, 'model_results/gradient_boosting_model.joblib')
    return metrics

def create_neural_network(input_dim):
    """Create a neural network model with residual connections for coordinate prediction."""
    def residual_block(x, units, dropout_rate=0.3):
        """Create a residual block with skip connection."""
        skip = x
        x = Dense(units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(units, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Add skip connection if dimensions match
        if K.int_shape(skip)[-1] == units:
            x = Add()([x, skip])
        return x
    
    inputs = Input(shape=(input_dim,))
    x = Dense(1024, activation='relu')(inputs)
    x = BatchNormalization()(x)
    
    # Residual blocks with decreasing units
    x = residual_block(x, 1024, dropout_rate=0.3)
    x = residual_block(x, 512, dropout_rate=0.3)
    x = residual_block(x, 256, dropout_rate=0.2)
    
    # Final layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(2, activation='linear')(x)  # Linear activation for regression
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use AMSGrad optimizer for better convergence
    optimizer = Adam(learning_rate=0.001, amsgrad=True)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model

def train_neural_network(X_train, X_test, y_train, y_test):
    """Train and evaluate Neural Network model with early stopping and checkpointing."""
    print("\nTraining Neural Network...")
    
    # Create model directory
    os.makedirs('model_results/checkpoints', exist_ok=True)
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'model_results/checkpoints/best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Create and train model
    model = create_neural_network(X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,  # Increased epochs since we have early stopping
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Neural Network")
    plot_predictions(y_test, y_pred, "Neural Network")
    
    # Plot training history
    plt.figure(figsize=(10, 4))
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
    plt.savefig('model_results/neural_network_training_history.png')
    plt.close()
    
    # Save model
    model.save('model_results/neural_network_model.keras')
    return metrics

def main():
    """Main function to train and evaluate all models."""
    # Create necessary directories
    os.makedirs('model_results', exist_ok=True)
    
    print("Loading preprocessed data...")
    X, y = load_preprocessed_data()
    
    # Additional feature scaling for neural network stability
    print("\nApplying additional feature scaling...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction
    print("\nApplying PCA dimensionality reduction...")
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X)
    print(f"Reduced features from {X.shape[1]} to {X_pca.shape[1]} components")
    
    print("\nCreating train-test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(X_pca, y)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create results directory
    os.makedirs('model_results', exist_ok=True)
    
    # Save PCA model
    joblib.dump(pca, 'model_results/pca_model.joblib')
    
    # Train and evaluate all models
    results = []
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=1000, max_depth=20, min_samples_split=5,
        min_samples_leaf=4, max_features='sqrt', bootstrap=True,
        oob_score=True, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results.append(evaluate_model(y_test, rf_pred, "Random Forest"))
    np.save('model_results/random_forest_predictions.npy', rf_pred)
    
    # Train Gradient Boosting
    gb_model = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.01, max_depth=6,
        min_samples_split=5, min_samples_leaf=4, subsample=0.8,
        validation_fraction=0.2, n_iter_no_change=10, random_state=42
    ), n_jobs=-1)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    results.append(evaluate_model(y_test, gb_pred, "Gradient Boosting"))
    np.save('model_results/gradient_boosting_predictions.npy', gb_pred)
    
    # Train Neural Network
    nn_model = create_neural_network(X_train.shape[1])
    nn_history = nn_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('model_results/neural_network_model.keras', save_best_only=True)
        ],
        verbose=1
    )
    nn_pred = nn_model.predict(X_test)
    results.append(evaluate_model(y_test, nn_pred, "Neural Network"))
    np.save('model_results/neural_network_predictions.npy', nn_pred)
    
    # Save test data for visualization
    np.save('model_results/test_actual.npy', y_test)
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_results/model_comparison.csv', index=False)
    print("\nResults and predictions saved to 'model_results' directory")

if __name__ == "__main__":
    main()
