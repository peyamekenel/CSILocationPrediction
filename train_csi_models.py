import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(features_path: str, location_info_path: str = "locations/location_coords.xlsx") -> tuple:
    """
    Load CSI features and location information for preprocessing.
    
    Args:
        features_path (str): Path to the npz file containing CSI features
        location_info_path (str): Path to Excel file with room coordinates
        
    Returns:
        tuple: (X, room_labels, space_ids, coordinates)
            - X: Combined features (flattened magnitudes and phases)
            - room_labels: Room numbers for classification
            - space_ids: Space IDs for potential coordinate mapping
            - coordinates: (x, y) coordinates for each room
    """
    # Load room coordinates with enhanced validation
    import pandas as pd
    try:
        coords_df = pd.read_excel(location_info_path)
        logger.info("\nCoordinate Data Summary:")
        logger.info(f"Total coordinate entries: {len(coords_df)}")
        logger.info(f"Columns present: {coords_df.columns.tolist()}")
        
        # Verify coordinate ranges
        logger.info(f"\nCoordinate Ranges:")
        logger.info(f"X range: [{coords_df['x_coord'].min():.2f}, {coords_df['x_coord'].max():.2f}]")
        logger.info(f"Y range: [{coords_df['y_coord'].min():.2f}, {coords_df['y_coord'].max():.2f}]")
        
        room_to_coords = {}
        invalid_rooms = []
        location_counts = {}
        
        for _, row in coords_df.iterrows():
            try:
                # Parse room number from location (e.g., "204_1" -> "204")
                location = str(row['location'])
                room_num = location.split('_')[0]
                x_coord = float(row['x_coord'])
                y_coord = float(row['y_coord'])
                
                if not (np.isfinite(x_coord) and np.isfinite(y_coord)):
                    invalid_rooms.append(f"{location} (invalid coordinates)")
                    continue
                
                # Keep track of how many times we've seen this room
                location_counts[room_num] = location_counts.get(room_num, 0) + 1
                
                # For rooms with multiple reference points, average their coordinates
                if room_num in room_to_coords:
                    old_x, old_y = room_to_coords[room_num]
                    count = location_counts[room_num]
                    # Update with running average
                    room_to_coords[room_num] = (
                        (old_x * (count - 1) + x_coord) / count,
                        (old_y * (count - 1) + y_coord) / count
                    )
                else:
                    room_to_coords[room_num] = (x_coord, y_coord)
                    
            except (ValueError, TypeError, IndexError) as e: 
                invalid_rooms.append(f"{location} (error: {str(e)})")
        
        logger.info(f"\nRoom Coordinate Mapping:")
        logger.info(f"Successfully mapped rooms: {len(room_to_coords)}")
        if invalid_rooms:
            logger.warning(f"Invalid room entries: {', '.join(invalid_rooms)}")
            
        # Log room reference point statistics
        logger.info("\nReference Point Statistics:")
        for room, count in sorted(location_counts.items()):
            logger.info(f"Room {room}: {count} reference points")
            
        # Verify coordinate distribution
        x_coords = np.array([coord[0] for coord in room_to_coords.values()])
        y_coords = np.array([coord[1] for coord in room_to_coords.values()])
        logger.info(f"\nMapped Coordinate Statistics:")
        logger.info(f"X mean: {np.mean(x_coords):.2f}, std: {np.std(x_coords):.2f}")
        logger.info(f"Y mean: {np.mean(y_coords):.2f}, std: {np.std(y_coords):.2f}")
        
    except Exception as e:
        logger.error(f"Could not load coordinates: {e}")
        logger.error("Traceback:", exc_info=True)
        room_to_coords = {}
    # Load and validate CSI features
    logger.info(f"\nLoading CSI features from: {features_path}")
    try:
        data = np.load(features_path, allow_pickle=True)
        magnitudes = data['magnitudes']
        phases = data['phases']
        room_numbers = np.array(data['room_numbers'])
        space_ids = data['space_ids'].astype(float)
        
        # Log feature shapes and basic statistics
        logger.info("\nFeature shapes:")
        logger.info(f"Magnitudes: {magnitudes.shape}")
        logger.info(f"Phases: {phases.shape}")
        logger.info(f"Number of rooms: {len(np.unique(room_numbers))}")
        logger.info(f"Number of samples: {len(room_numbers)}")
        
        # Verify data integrity
        if len(magnitudes) != len(room_numbers):
            raise ValueError(f"Mismatched samples: {len(magnitudes)} magnitude samples but {len(room_numbers)} labels")
        if len(phases) != len(room_numbers):
            raise ValueError(f"Mismatched samples: {len(phases)} phase samples but {len(room_numbers)} labels")
            
    except Exception as e:
        logger.error(f"Error loading CSI features: {str(e)}")
        raise
    
    # Calculate comprehensive CSI statistics
    logger.info("\nCalculating CSI statistics...")
    
    # Magnitude features
    mag_mean = np.mean(magnitudes, axis=1)  # (samples, subcarriers)
    mag_std = np.std(magnitudes, axis=1)
    mag_max = np.max(magnitudes, axis=1)
    mag_min = np.min(magnitudes, axis=1)
    mag_median = np.median(magnitudes, axis=1)
    mag_skew = np.mean(((magnitudes - np.mean(magnitudes, axis=1, keepdims=True)) / 
                       np.std(magnitudes, axis=1, keepdims=True)) ** 3, axis=1)
    
    # Phase features with unwrapping
    unwrapped_phases = np.unwrap(phases, axis=1)  # Unwrap along packet dimension
    phase_mean = np.mean(unwrapped_phases, axis=1)
    phase_std = np.std(unwrapped_phases, axis=1)
    phase_max = np.max(unwrapped_phases, axis=1)
    phase_min = np.min(unwrapped_phases, axis=1)
    phase_range = phase_max - phase_min
    
    # Log feature statistics
    logger.info("\nCSI Feature Statistics:")
    logger.info(f"Magnitude - Mean range: [{np.min(mag_mean):.2f}, {np.max(mag_mean):.2f}]")
    logger.info(f"Magnitude - Std range: [{np.min(mag_std):.2f}, {np.max(mag_std):.2f}]")
    logger.info(f"Phase - Mean range: [{np.min(phase_mean):.2f}, {np.max(phase_mean):.2f}]")
    logger.info(f"Phase - Std range: [{np.min(phase_std):.2f}, {np.max(phase_std):.2f}]")
    
    # Add spatial features if coordinates are available
    coordinates = []
    for room in room_numbers:
        if room in room_to_coords:
            coordinates.append(room_to_coords[room])
        else:
            coordinates.append((0, 0))  # Default coordinates if not found
    coordinates = np.array(coordinates)
    
    # Group rooms into zones based on coordinates with enhanced clustering
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Enhanced zone clustering with better initialization and parameters
    n_zones = 5
    
    # Use MinMaxScaler for better spatial separation
    from sklearn.preprocessing import MinMaxScaler
    
    x_coords = coordinates[:, 0].reshape(-1, 1)
    y_coords = coordinates[:, 1].reshape(-1, 1)
    
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    scaled_x = x_scaler.fit_transform(x_coords)
    scaled_y = y_scaler.fit_transform(y_coords)
    
    # Add slight noise to prevent duplicate points
    np.random.seed(42)
    scaled_x += np.random.normal(0, 0.001, scaled_x.shape)
    scaled_y += np.random.normal(0, 0.001, scaled_y.shape)
    
    # Combine scaled coordinates
    scaled_coordinates = np.hstack([scaled_x, scaled_y])
    
    # Log coordinate statistics before clustering
    logger.info("\nCoordinate Statistics before clustering:")
    logger.info(f"X range: [{np.min(x_coords):.2f}, {np.max(x_coords):.2f}]")
    logger.info(f"Y range: [{np.min(y_coords):.2f}, {np.max(y_coords):.2f}]")
    logger.info(f"Scaled X range: [{np.min(scaled_x):.3f}, {np.max(scaled_x):.3f}]")
    logger.info(f"Scaled Y range: [{np.min(scaled_y):.3f}, {np.max(scaled_y):.3f}]")
    
    # Calculate optimal number of clusters using silhouette score
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    inertias = []
    cluster_range = range(2, 7)
    
    for n_clusters in cluster_range:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=500,
            tol=1e-6
        )
        cluster_labels = kmeans.fit_predict(scaled_coordinates)
        silhouette_avg = silhouette_score(scaled_coordinates, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertias.append(kmeans.inertia_)
        logger.info(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}")
    
    # Choose optimal number of clusters
    optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    logger.info(f"\nOptimal number of clusters: {optimal_n_clusters}")
    
    # Initialize KMeans with optimal parameters
    zone_clustering = KMeans(
        n_clusters=optimal_n_clusters,
        random_state=42,
        n_init=10,
        max_iter=500,
        tol=1e-6
    )
    
    # Fit and predict zones
    room_zones = zone_clustering.fit_predict(scaled_coordinates)
    
    # Calculate clustering quality metrics
    if len(np.unique(room_zones)) > 1:  # Only calculate if we have more than one cluster
        silhouette_avg = silhouette_score(scaled_coordinates, room_zones)
        logger.info(f"\nClustering Quality:")
        logger.info(f"Silhouette Score: {silhouette_avg:.3f}")
        logger.info(f"Inertia: {zone_clustering.inertia_:.3f}")
    
    # Log zone distribution with enhanced statistics
    logger.info("\nZone distribution:")
    for zone in range(n_zones):
        zone_mask = room_zones == zone
        zone_rooms = room_numbers[zone_mask]
        unique_zone_rooms = np.unique(zone_rooms)
        
        # Sort rooms for consistent display
        sorted_rooms = sorted(unique_zone_rooms)
        room_list = ', '.join(sorted_rooms[:5])
        if len(sorted_rooms) > 5:
            room_list += '...'
            
        logger.info(f"\nZone {zone}:")
        logger.info(f"  Total samples: {np.sum(zone_mask)}")
        logger.info(f"  Unique rooms: {len(unique_zone_rooms)}")
        logger.info(f"  Rooms: {room_list}")
        
        # Log sample distribution within zone
        logger.info("  Sample distribution:")
        for room in sorted_rooms:
            room_count = np.sum((room_zones == zone) & (room_numbers == room))
            logger.info(f"    Room {room}: {room_count} samples")
    
    # Add zone information to features with reduced dimensionality
    zone_features = np.zeros((len(room_numbers), n_zones))
    zone_features[np.arange(len(room_numbers)), room_zones] = 1
    
    # Advanced features
    # Subcarrier correlation (between adjacent subcarriers)
    mag_corr = np.zeros((magnitudes.shape[0], magnitudes.shape[2]-1))
    phase_corr = np.zeros((phases.shape[0], phases.shape[2]-1))
    
    for i in range(magnitudes.shape[0]):
        for j in range(magnitudes.shape[2]-1):
            try:
                # Handle potential NaN values in correlation calculation
                mag_corr_matrix = np.corrcoef(magnitudes[i,:,j], magnitudes[i,:,j+1])
                phase_corr_matrix = np.corrcoef(phases[i,:,j], phases[i,:,j+1])
                
                mag_corr[i,j] = 0.0 if np.isnan(mag_corr_matrix[0,1]) else mag_corr_matrix[0,1]
                phase_corr[i,j] = 0.0 if np.isnan(phase_corr_matrix[0,1]) else phase_corr_matrix[0,1]
            except Exception as e:
                logger.warning(f"Error calculating correlation for sample {i}, subcarrier {j}: {str(e)}")
                mag_corr[i,j] = 0.0
                phase_corr[i,j] = 0.0
    
    # Phase differences between adjacent subcarriers
    phase_diff = np.diff(phases, axis=2)
    phase_diff_mean = np.mean(phase_diff, axis=1)
    phase_diff_std = np.std(phase_diff, axis=1)
    
    # Magnitude ratios between adjacent subcarriers with better handling of edge cases
    mag_ratio = np.zeros_like(magnitudes[:,:,1:])
    denominator = magnitudes[:,:,:-1]
    # Avoid division by zero and handle edge cases
    mask = denominator > 1e-10
    mag_ratio[mask] = magnitudes[:,:,1:][mask] / denominator[mask]
    mag_ratio_mean = np.mean(mag_ratio, axis=1)
    mag_ratio_std = np.std(mag_ratio, axis=1)
    
    # Combine all features with enhanced CSI statistics and zone information
    X = np.hstack([
        # Enhanced CSI magnitude statistics
        mag_mean, mag_std, mag_max, mag_min, mag_median, mag_skew,
        # Enhanced CSI phase statistics
        phase_mean, phase_std, phase_max, phase_min, phase_range,
        # Advanced correlation features
        mag_corr, phase_corr,
        phase_diff_mean,
        # Spatial context features
        coordinates,
        zone_features
    ])
    
    # Log combined feature information
    logger.info("\nCombined Feature Information:")
    logger.info(f"Total features: {X.shape[1]}")
    logger.info("Feature groups:")
    logger.info(f"- Magnitude statistics: {mag_mean.shape[1] * 6} features")
    logger.info(f"- Phase statistics: {phase_mean.shape[1] * 5} features")
    logger.info(f"- Correlation features: {mag_corr.shape[1] + phase_corr.shape[1]} features")
    logger.info(f"- Phase difference features: {phase_diff_mean.shape[1]} features")
    logger.info(f"- Spatial features: {coordinates.shape[1]} features")
    logger.info(f"- Zone features: {zone_features.shape[1]} features")
    
    # Log feature statistics and distributions
    logger.info(f"\nPreprocessed features shape: {X.shape}")
    logger.info("\nFeature Statistics:")
    logger.info(f"Magnitude features - Mean: {np.mean(mag_mean):.3f}, Std: {np.std(mag_mean):.3f}")
    logger.info(f"Phase features - Mean: {np.mean(phase_mean):.3f}, Std: {np.std(phase_mean):.3f}")
    logger.info(f"Correlation features - Mean: {np.mean(mag_corr):.3f}, Std: {np.std(mag_corr):.3f}")
    
    # Log unique room numbers and their counts
    unique_rooms, room_counts = np.unique(room_numbers, return_counts=True)
    logger.info("\nRoom distribution:")
    for room, count in zip(unique_rooms, room_counts):
        logger.info(f"Room {room}: {count} samples")
    
    # Validate feature alignment
    if len(room_numbers) != X.shape[0]:
        raise ValueError(f"Misaligned features and labels: {X.shape[0]} samples but {len(room_numbers)} room labels")
    if len(space_ids) != X.shape[0]:
        raise ValueError(f"Misaligned features and space IDs: {X.shape[0]} samples but {len(space_ids)} space IDs")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.warning("Warning: Features contain NaN or infinite values!")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info("Replaced NaN and infinite values with 0.0")
    
    return X, room_numbers, space_ids, coordinates, room_zones

def train_room_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_zones: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_zones: np.ndarray
) -> tuple:
    """
    Train a random forest classifier for room prediction using zone information.
    
    Args:
        X_train, X_val: Training and validation features
        y_train, y_val: Room number labels
        train_zones, val_zones: Zone information for spatial context
        
    Returns:
        tuple: (trained_model, scaler, selector, pca, label_encoder, validation_accuracy)
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Convert string labels to integers
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    
    # Create zone-based features
    n_zones = len(np.unique(np.concatenate([train_zones, val_zones])))
    train_zone_features = np.zeros((len(train_zones), n_zones))
    val_zone_features = np.zeros((len(val_zones), n_zones))
    train_zone_features[np.arange(len(train_zones)), train_zones] = 1
    val_zone_features[np.arange(len(val_zones)), val_zones] = 1
    
    # Combine original features with zone features
    X_train_with_zones = np.hstack([X_train, train_zone_features])
    X_val_with_zones = np.hstack([X_val, val_zone_features])
    
    # Enhanced feature engineering with zone awareness
    
    # 1. Remove constant and near-constant features
    selector = VarianceThreshold(threshold=1e-5)
    X_train_selected = selector.fit_transform(X_train_with_zones)
    X_val_selected = selector.transform(X_val_with_zones)
    
    logger.info(f"\nInitial feature selection: {X_train_selected.shape[1]} features retained out of {X_train_with_zones.shape[1]}")
    logger.info(f"Zone features start index: {X_train.shape[1]}")
    
    # 2. Scale features (excluding one-hot encoded zone features)
    scaler = StandardScaler()
    # Scale only the CSI features, not the zone features
    n_csi_features = X_train.shape[1]
    X_train_csi = X_train_selected[:, :n_csi_features]
    X_val_csi = X_val_selected[:, :n_csi_features]
    X_train_zones = X_train_selected[:, n_csi_features:]
    X_val_zones = X_val_selected[:, n_csi_features:]
    
    # Scale CSI features
    X_train_csi_scaled = scaler.fit_transform(X_train_csi)
    X_val_csi_scaled = scaler.transform(X_val_csi)
    
    # Recombine with unscaled zone features
    X_train_scaled = np.hstack([X_train_csi_scaled, X_train_zones])
    X_val_scaled = np.hstack([X_val_csi_scaled, X_val_zones])
    
    # 3. Apply PCA with moderate variance retention
    pca = PCA(n_components=0.95)  # Keep 95% of variance to reduce noise
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    # Log retained components
    logger.info(f"\nPCA components retained: {X_train_pca.shape[1]}")
    logger.info(f"Cumulative explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.close()
    
    logger.info(f"\nPCA components retained: {X_train_pca.shape[1]}")
    logger.info(f"PCA explained variance ratio (first 5 components): {pca.explained_variance_ratio_[:5]}")
    logger.info(f"Total explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Extract building section from room numbers (e.g., '204' -> '2', 'A203' -> 'A2')
    def get_section(room):
        if room.startswith('A'):
            return 'A' + room[1]
        return room[0]
    
    # Enhanced data augmentation
    X_train_aug = []
    y_train_aug = []
    
    for i in range(len(X_train_pca)):
        # Original sample
        X_train_aug.append(X_train_pca[i])
        y_train_aug.append(y_train_encoded[i])
        
        # Add multiple perturbed versions with varying noise levels
        for noise_scale in [0.01, 0.02, 0.03]:  # Multiple noise scales
            for _ in range(3):  # 3 samples per noise scale
                noise = np.random.normal(0, noise_scale, X_train_pca[i].shape)
                X_train_aug.append(X_train_pca[i] + noise)
                y_train_aug.append(y_train_encoded[i])
    
    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)
    
    logger.info(f"\nAugmented training set size: {len(X_train_aug)} (original: {len(X_train_pca)})")
    
    # Train a single Random Forest classifier with optimized parameters
    section_clf = RandomForestClassifier(
        n_estimators=500,  # More trees for better generalization
        max_depth=15,      # Limit depth to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',  # Better handling of imbalanced classes
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    # Train Random Forest with zone-aware parameters
    room_clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,  # Allow deeper trees to capture zone relationships
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced_subsample',  # Better handling of imbalanced zones
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,  # Enable out-of-bag score estimation
        n_jobs=-1,
        random_state=42
    )
    
    # Train on augmented data
    room_clf.fit(X_train_aug, y_train_aug)
    
    # Analyze feature importance with zone context
    importances = room_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    logger.info("\nTop 15 most important features:")
    for f in range(min(15, len(indices))):
        feature_type = "Zone" if indices[f] >= n_csi_features else "CSI"
        logger.info(f"{f + 1}. {feature_type} Feature {indices[f]}: {importances[indices[f]]:.4f}")
    
    # Predictions and confidence analysis by zone
    y_pred_encoded = room_clf.predict(X_val_pca)
    y_pred_proba = room_clf.predict_proba(X_val_pca)
    confidence_scores = np.max(y_pred_proba, axis=1)
    
    # Analyze performance by zone
    logger.info("\nPerformance analysis by zone:")
    for zone in np.unique(val_zones):
        zone_mask = val_zones == zone
        if np.sum(zone_mask) > 0:
            zone_accuracy = accuracy_score(
                y_val_encoded[zone_mask],
                y_pred_encoded[zone_mask]
            )
            zone_confidence = confidence_scores[zone_mask]
            
            logger.info(f"\nZone {zone}:")
            logger.info(f"Accuracy: {zone_accuracy:.4f}")
            logger.info(f"Mean confidence: {np.mean(zone_confidence):.4f}")
            logger.info(f"Median confidence: {np.median(zone_confidence):.4f}")
    
    # Overall performance
    y_pred = le.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info("\nOverall Performance:")
    logger.info(f"Out-of-bag score: {room_clf.oob_score_:.4f}")
    logger.info("\nRoom Classification Report:")
    logger.info(classification_report(y_val, y_pred))
    
    return room_clf, scaler, selector, pca, le, accuracy

def train_coordinate_regressor(
    X_train: np.ndarray,
    space_ids_train: np.ndarray,
    train_zones: np.ndarray,
    X_val: np.ndarray,
    space_ids_val: np.ndarray,
    val_zones: np.ndarray
) -> tuple:
    """
    Train a random forest regressor for space ID prediction with zone awareness.
    
    Args:
        X_train, X_val: Training and validation features
        space_ids_train, space_ids_val: Space ID labels
        train_zones, val_zones: Zone information for spatial context
        
    Returns:
        tuple: (trained_model, scaler, validation_mse)
    """
    # Create zone-based features
    n_zones = len(np.unique(np.concatenate([train_zones, val_zones])))
    train_zone_features = np.zeros((len(train_zones), n_zones))
    val_zone_features = np.zeros((len(val_zones), n_zones))
    train_zone_features[np.arange(len(train_zones)), train_zones] = 1
    val_zone_features[np.arange(len(val_zones)), val_zones] = 1
    
    # Combine original features with zone features
    X_train_with_zones = np.hstack([X_train, train_zone_features])
    X_val_with_zones = np.hstack([X_val, val_zone_features])
    
    # Log feature dimensions
    logger.info("\nRegressor Feature Dimensions:")
    logger.info(f"CSI features: {X_train.shape[1]}")
    logger.info(f"Zone features: {n_zones}")
    logger.info(f"Total features: {X_train_with_zones.shape[1]}")
    
    # Scale features (excluding one-hot encoded zone features)
    scaler = StandardScaler()
    n_csi_features = X_train.shape[1]
    
    # Split features
    X_train_csi = X_train_with_zones[:, :n_csi_features]
    X_val_csi = X_val_with_zones[:, :n_csi_features]
    X_train_zones = X_train_with_zones[:, n_csi_features:]
    X_val_zones = X_val_with_zones[:, n_csi_features:]
    
    # Scale CSI features
    X_train_csi_scaled = scaler.fit_transform(X_train_csi)
    X_val_csi_scaled = scaler.transform(X_val_csi)
    
    # Recombine with unscaled zone features
    X_train_scaled = np.hstack([X_train_csi_scaled, X_train_zones])
    X_val_scaled = np.hstack([X_val_csi_scaled, X_val_zones])
    
    # Train zone-aware regressor
    reg = RandomForestRegressor(
        n_estimators=500,  # Increased trees for better generalization
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    # Verify feature alignment
    if len(space_ids_train) != X_train_scaled.shape[0]:
        raise ValueError(f"Training feature-label mismatch: {X_train_scaled.shape[0]} samples but {len(space_ids_train)} labels")
    if len(space_ids_val) != X_val_scaled.shape[0]:
        raise ValueError(f"Validation feature-label mismatch: {X_val_scaled.shape[0]} samples but {len(space_ids_val)} labels")
    
    reg.fit(X_train_scaled, space_ids_train)
    
    # Evaluate overall and by zone
    y_pred = reg.predict(X_val_scaled)
    mse = mean_squared_error(space_ids_val, y_pred)
    
    logger.info("\nSpace ID Regression Results:")
    logger.info(f"Overall MSE: {mse:.4f}")
    logger.info(f"Overall RMSE: {np.sqrt(mse):.4f}")
    
    # Analyze performance by zone
    logger.info("\nRegression performance by zone:")
    for zone in np.unique(val_zones):
        zone_mask = val_zones == zone
        if np.sum(zone_mask) > 0:
            zone_mse = mean_squared_error(
                space_ids_val[zone_mask],
                y_pred[zone_mask]
            )
            logger.info(f"Zone {zone}:")
            logger.info(f"  MSE: {zone_mse:.4f}")
            logger.info(f"  RMSE: {np.sqrt(zone_mse):.4f}")
            logger.info(f"  Samples: {np.sum(zone_mask)}")
    
    return reg, scaler, mse

if __name__ == "__main__":
    # Load reference data for training with spatial information
    ref_features_path = "processed_features/reference_features.npz"
    location_info_path = "locations/location_coords.xlsx"
    X_ref, rooms_ref, spaces_ref, coordinates_ref, zones_ref = load_and_preprocess_data(
        ref_features_path,
        location_info_path
    )
    
    # Ensure consistent room number encoding
    rooms_ref = np.array([str(r) for r in rooms_ref])
    
    # Create balanced train/validation split within each zone
    train_indices = []
    val_indices = []
    
    # Sort unique rooms for consistent splitting
    unique_rooms = np.unique(rooms_ref)
    unique_rooms.sort()
    
    # Split rooms within each zone to maintain spatial distribution
    for zone in range(5):  # Using 5 zones as configured
        zone_mask = zones_ref == zone
        zone_rooms = np.unique(rooms_ref[zone_mask])
        
        # For each room in the zone
        for room in zone_rooms:
            room_indices = np.where((zones_ref == zone) & (rooms_ref == room))[0]
            n_samples = len(room_indices)
            
            if n_samples == 0:
                continue
                
            # Split samples for this room (60-40)
            n_train = max(1, int(0.6 * n_samples))  # At least 1 sample for training
            
            # Randomly shuffle indices
            np.random.seed(42 + hash(str(zone) + room))  # Different seed for each room
            np.random.shuffle(room_indices)
            
            # Add to train/val sets
            train_indices.extend(room_indices[:n_train])
            val_indices.extend(room_indices[n_train:])
    
    # Convert to arrays and create masks
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    
    # Split features and labels using indices
    X_train = X_ref[train_indices]
    X_val = X_ref[val_indices]
    y_train = rooms_ref[train_indices]
    y_val = rooms_ref[val_indices]
    spaces_train = spaces_ref[train_indices]
    spaces_val = spaces_ref[val_indices]
    zones_train = zones_ref[train_indices]
    zones_val = zones_ref[val_indices]
    
    # Log split information with zone details
    logger.info("\nZone-based split statistics:")
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Training zones: {len(np.unique(zones_train))} zones")
    logger.info(f"Validation zones: {len(np.unique(zones_val))} zones")
    logger.info(f"Unique rooms in training: {len(np.unique(y_train))}")
    logger.info(f"Unique rooms in validation: {len(np.unique(y_val))}")
    
    # Print detailed zone distribution in split
    logger.info("\nZone distribution in split:")
    for zone in range(5):  # Using 5 zones as configured
        train_count = np.sum(zones_train == zone)
        val_count = np.sum(zones_val == zone)
        
        # Get unique rooms in each split for this zone
        train_zone_rooms = np.unique(y_train[zones_train == zone])
        val_zone_rooms = np.unique(y_val[zones_val == zone])
        
        logger.info(f"\nZone {zone}:")
        logger.info(f"  Training: {train_count} samples, {len(train_zone_rooms)} rooms")
        logger.info(f"  Validation: {val_count} samples, {len(val_zone_rooms)} rooms")
        logger.info(f"  Training rooms: {', '.join(sorted(train_zone_rooms)[:5])}{'...' if len(train_zone_rooms) > 5 else ''}")
        logger.info(f"  Validation rooms: {', '.join(sorted(val_zone_rooms)[:5])}{'...' if len(val_zone_rooms) > 5 else ''}")
        
        # Log sample distribution within zone
        logger.info("  Sample distribution:")
        for room in sorted(set(train_zone_rooms) | set(val_zone_rooms)):
            train_room_count = np.sum((zones_train == zone) & (y_train == room))
            val_room_count = np.sum((zones_val == zone) & (y_val == room))
            logger.info(f"    Room {room}: {train_room_count} train, {val_room_count} validation samples")
    
    # Train and evaluate room classifier with zone information
    logger.info("\nTraining room classifier...")
    room_clf, room_scaler, selector, pca, label_encoder, room_accuracy = train_room_classifier(
        X_train, y_train, zones_train,
        X_val, y_val, zones_val
    )
    
    # Train and evaluate coordinate regressor with zone information
    logger.info("\nTraining space ID regressor...")
    space_reg, space_scaler, space_mse = train_coordinate_regressor(
        X_train, spaces_train, zones_train,
        X_val, spaces_val, zones_val
    )
    
    # Save models and scalers
    os.makedirs("models", exist_ok=True)
    np.save("models/room_scaler.npy", {
        'mean': room_scaler.mean_,
        'scale': room_scaler.scale_
    })
    np.save("models/space_scaler.npy", {
        'mean': space_scaler.mean_,
        'scale': space_scaler.scale_
    })
    
    # Load test data and evaluate with spatial information
    test_features_path = "processed_features/test_features.npz"
    X_test, rooms_test, spaces_test, coordinates_test, zones_test = load_and_preprocess_data(
        test_features_path,
        location_info_path
    )
    
    # Apply feature selection, scaling, and PCA to test data
    X_test_selected = selector.transform(X_test)
    X_test_room = room_scaler.transform(X_test_selected)
    X_test_room = pca.transform(X_test_room)
    X_test_space = space_scaler.transform(X_test)
    
    # Evaluate on test set
    rooms_test_encoded = label_encoder.transform(rooms_test)
    # Direct prediction on test set
    room_pred_encoded = room_clf.predict(X_test_room)
    
    # Get prediction probabilities for confidence analysis
    test_pred_proba = room_clf.predict_proba(X_test_room)
    test_confidence_scores = np.max(test_pred_proba, axis=1)
    
    logger.info("\nTest Set Confidence Statistics:")
    logger.info(f"Mean confidence: {np.mean(test_confidence_scores):.4f}")
    logger.info(f"Median confidence: {np.median(test_confidence_scores):.4f}")
    logger.info(f"Min confidence: {np.min(test_confidence_scores):.4f}")
    logger.info(f"Max confidence: {np.max(test_confidence_scores):.4f}")
    room_pred = label_encoder.inverse_transform(room_pred_encoded)
    space_pred = space_reg.predict(X_test_space)
    
    logger.info("\nTest Set Results:")
    logger.info("Room Classification Report:")
    logger.info(classification_report(rooms_test, room_pred))
    
    test_mse = mean_squared_error(spaces_test, space_pred)
    logger.info(f"Space ID Test MSE: {test_mse:.4f}")
    logger.info(f"Space ID Test RMSE: {np.sqrt(test_mse):.4f}")
