import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_zone_distribution():
    # Load coordinates
    coords_df = pd.read_excel('locations/location_coords.xlsx')
    
    # Extract room numbers and coordinates
    coords_df['room'] = coords_df['location'].apply(lambda x: x.split('_')[0])
    X = coords_df[['x_coord', 'y_coord']].values
    
    # Scale coordinates
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    X_scaled = np.column_stack([
        x_scaler.fit_transform(X[:, 0].reshape(-1, 1)),
        y_scaler.fit_transform(X[:, 1].reshape(-1, 1))
    ])
    
    # Add small noise to prevent duplicates
    np.random.seed(42)
    X_scaled += np.random.normal(0, 0.001, X_scaled.shape)
    
    # Perform clustering
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Calculate silhouette scores
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
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertias.append(kmeans.inertia_)
        logger.info(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}")
    
    # Choose optimal number of clusters
    optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    logger.info(f"\nOptimal number of clusters: {optimal_n_clusters}")
    
    # Perform final clustering
    kmeans = KMeans(
        n_clusters=optimal_n_clusters,
        random_state=42,
        n_init=10,
        max_iter=500,
        tol=1e-6
    )
    zone_labels = kmeans.fit_predict(X_scaled)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot points colored by zone
    scatter = plt.scatter(X[:, 0], X[:, 1], c=zone_labels, cmap='viridis', alpha=0.6)
    
    # Add room numbers as labels
    for i, room in enumerate(coords_df['room']):
        plt.annotate(room, (X[i, 0], X[i, 1]), fontsize=8)
    
    plt.colorbar(scatter, label='Zone')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('Room Locations Colored by Zone')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('visualizations/zone_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("Zone distribution visualization saved to visualizations/zone_distribution.png")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(list(cluster_range), silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/silhouette_scores.png', dpi=300, bbox_inches='tight')
    logger.info("Silhouette scores visualization saved to visualizations/silhouette_scores.png")

if __name__ == "__main__":
    visualize_zone_distribution()
