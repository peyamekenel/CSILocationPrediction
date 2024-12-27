import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle
import os
import matplotlib.pyplot as plt

def load_data():
    """Load PCA-reduced features and labels."""
    # Load features
    X_train = pd.read_csv('reduced_data/features_train_pca.csv')
    X_val = pd.read_csv('reduced_data/features_validation_pca.csv')
    X_test = pd.read_csv('reduced_data/features_test_pca.csv')
    
    # Load labels
    y_train = pd.read_csv('cleaned_data/labels_train.csv')
    y_val = pd.read_csv('cleaned_data/labels_validation.csv')
    y_test = pd.read_csv('cleaned_data/labels_test.csv')
    
    # Convert room labels to sequential classes (0-N)
    unique_rooms = sorted(pd.concat([y_train['room'], y_val['room'], y_test['room']]).unique())
    room_to_class = {room: idx for idx, room in enumerate(unique_rooms)}
    
    y_train['room_class'] = y_train['room'].map(room_to_class)
    y_val['room_class'] = y_val['room'].map(room_to_class)
    y_test['room_class'] = y_test['room'].map(room_to_class)
    
    # Convert y coordinate to binary class (0 for 4.48, 1 for 5.60)
    y_train['y_class'] = (y_train['y'] > 5.0).astype(int)
    y_val['y_class'] = (y_val['y'] > 5.0).astype(int)
    y_test['y_class'] = (y_test['y'] > 5.0).astype(int)
    
    # Print data information
    print("\nData shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    
    print("\nRoom class distribution (training set):")
    print(y_train['room_class'].value_counts())
    
    print("\nY coordinate class distribution (training set):")
    print(y_train['y_class'].value_counts())
    
    return X_train, X_val, X_test, y_train, y_val, y_test, room_to_class

def train_zone_classifier(X_train, y_train, X_val, y_val, room_to_class):
    """Train and evaluate multiple classifiers for zone prediction."""
    # Extract room/zone labels
    y_train_zone = y_train['room_class']
    y_val_zone = y_val['room_class']
    
    # Calculate class weights for all possible classes
    all_classes = list(range(len(room_to_class)))
    class_counts = pd.Series(y_train_zone).value_counts()
    
    # Ensure all classes have a weight, even if not in training set
    weights = []
    n_samples = len(y_train_zone)
    n_classes = len(all_classes)
    
    for class_idx in all_classes:
        if class_idx in class_counts.index:
            weights.append(n_samples / (n_classes * class_counts[class_idx]))
        else:
            # For classes not in training set, use the average weight
            weights.append(1.0)
    
    class_weights = dict(zip(all_classes, weights))
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define classifiers with class weights and appropriate parameters for small dataset
    classifiers = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight=class_weights,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight=class_weights,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=3,
            weights='distance'
        ),
        'NeuralNet': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
    }
    
    # Also train a classifier for y-coordinate binary classification
    y_classifiers = {
        'RandomForest_Y': RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_accuracy = float('-inf')  # Initialize to negative infinity to ensure we always get a best model
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name} classifier...")
        
        # Train classifier
        clf.fit(X_train_scaled, y_train_zone)
        
        # Evaluate on validation set
        y_pred = clf.predict(X_val_scaled)
        accuracy = accuracy_score(y_val_zone, y_pred)
        
        # Save results
        results[name] = {
            'accuracy': accuracy,
            'report': classification_report(y_val_zone, y_pred)
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Track best model (will always update on first iteration)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_model = clf
        
        print(f"\n{name} Classifier Results:")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(results[name]['report'])
    
    # Ensure we have a model, even if all accuracies are 0
    if best_model is None:
        print("\nWarning: No model performed well. Using RandomForest as fallback.")
        best_model = classifiers['RandomForest']
    
    return best_model, results

def train_coordinate_regressor(X_train, y_train, X_val, y_val):
    """Train and evaluate regressor for x-coordinate prediction and classifier for y-coordinate."""
    # Extract x coordinates (regression target)
    y_train_x = y_train['x']
    y_val_x = y_val['x']
    
    # Extract y coordinates (binary classification target)
    y_train_y = y_train['y_class']
    y_val_y = y_val['y_class']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train Random Forest Regressor for x coordinate
    rf_regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    rf_regressor.fit(X_train_scaled, y_train_x)
    
    # Train Random Forest Classifier for y coordinate
    rf_classifier_y = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    rf_classifier_y.fit(X_train_scaled, y_train_y)
    
    # Evaluate x coordinate regression
    x_pred = rf_regressor.predict(X_val_scaled)
    mse_x = mean_squared_error(y_val_x, x_pred)
    r2_x = r2_score(y_val_x, x_pred)
    
    print("\nX Coordinate Regression Results:")
    print(f"Mean Squared Error: {mse_x:.4f}")
    print(f"R² Score: {r2_x:.4f}")
    
    # Evaluate y coordinate classification
    y_pred_class = rf_classifier_y.predict(X_val_scaled)
    y_accuracy = accuracy_score(y_val_y, y_pred_class)
    
    print("\nY Coordinate Classification Results:")
    print(f"Accuracy: {y_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val_y, y_pred_class))
    
    # Visualize predictions vs actual
    plt.figure(figsize=(10, 6))
    # Convert y_pred_class back to actual y values
    y_pred_values = np.where(y_pred_class == 0, 4.480563296109769, 5.600704120137212)
    plt.scatter(y_val_x, y_val['y'], c='blue', label='Actual')
    plt.scatter(x_pred, y_pred_values, c='red', label='Predicted')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted vs Actual Coordinates')
    plt.legend()
    plt.savefig('coordinate_predictions.png')
    plt.close()
    
    return (rf_regressor, rf_classifier_y), {'mse_x': mse_x, 'r2_x': r2_x, 'y_accuracy': y_accuracy}

def save_models(zone_classifier, coordinate_models, room_to_class):
    """Save trained models and mappings to disk."""
    os.makedirs('models', exist_ok=True)
    
    # Save zone classifier
    with open('models/zone_classifier.pkl', 'wb') as f:
        pickle.dump(zone_classifier, f)
    
    # Save coordinate prediction models
    x_regressor, y_classifier = coordinate_models
    with open('models/x_coordinate_regressor.pkl', 'wb') as f:
        pickle.dump(x_regressor, f)
    with open('models/y_coordinate_classifier.pkl', 'wb') as f:
        pickle.dump(y_classifier, f)
        
    # Save room mapping for later use
    with open('models/room_mapping.pkl', 'wb') as f:
        pickle.dump(room_to_class, f)

def main():
    # Load data
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, room_to_class = load_data()
    
    # Train zone classifier
    print("\nTraining zone classifiers...")
    best_classifier, classification_results = train_zone_classifier(X_train, y_train, X_val, y_val, room_to_class)
    
    # Train coordinate regressor
    print("\nTraining coordinate regressor...")
    coordinate_models, regression_results = train_coordinate_regressor(X_train, y_train, X_val, y_val)
    
    # Save models
    print("\nSaving models...")
    save_models(best_classifier, coordinate_models, room_to_class)
    
    print("\nTraining complete! Models saved in 'models' directory.")

if __name__ == "__main__":
    main()
