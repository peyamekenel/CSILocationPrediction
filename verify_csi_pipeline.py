import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from train_csi_models import load_and_preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_csi_data(magnitudes, phases, room_number, save_dir='visualizations'):
    """Plot magnitude and phase data for visual verification."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot magnitudes
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(magnitudes.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title(f'CSI Magnitudes - Room {room_number}')
    plt.xlabel('Packet Index')
    plt.ylabel('Subcarrier Index')
    
    # Plot phases
    plt.subplot(1, 2, 2)
    plt.imshow(phases.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Phase')
    plt.title(f'CSI Phases - Room {room_number}')
    plt.xlabel('Packet Index')
    plt.ylabel('Subcarrier Index')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/csi_visualization_room_{room_number}.png')
    plt.close()

def verify_packet_consistency(features_path):
    """Verify packet count consistency across samples."""
    data = np.load(features_path, allow_pickle=True)
    magnitudes = data['magnitudes']
    phases = data['phases']
    room_numbers = data['room_numbers']
    
    logger.info("\nPacket Consistency Check:")
    logger.info(f"Number of samples: {len(room_numbers)}")
    logger.info(f"Magnitude shape: {magnitudes.shape}")
    logger.info(f"Phase shape: {phases.shape}")
    
    # Check packet counts per room
    unique_rooms = np.unique(room_numbers)
    for room in unique_rooms:
        room_mask = room_numbers == room
        room_magnitudes = magnitudes[room_mask]
        logger.info(f"\nRoom {room}:")
        logger.info(f"  Number of samples: {len(room_magnitudes)}")
        logger.info(f"  Packets per sample: {room_magnitudes.shape[1]}")
        logger.info(f"  Subcarriers: {room_magnitudes.shape[2]}")

def main():
    # Load reference data
    ref_features_path = "processed_features/reference_features.npz"
    location_info_path = "locations/location_coords.xlsx"
    
    # Load and preprocess data
    X, rooms, spaces, coordinates, zones = load_and_preprocess_data(
        ref_features_path,
        location_info_path
    )
    
    # Verify packet consistency
    logger.info("\nVerifying packet consistency...")
    verify_packet_consistency(ref_features_path)
    
    # Load raw data for visualization
    data = np.load(ref_features_path, allow_pickle=True)
    magnitudes = data['magnitudes']
    phases = data['phases']
    room_numbers = data['room_numbers']
    
    # Plot CSI data for first sample of each unique room
    logger.info("\nGenerating CSI visualizations...")
    unique_rooms = np.unique(room_numbers)
    for room in unique_rooms[:5]:  # Plot first 5 rooms
        room_idx = np.where(room_numbers == room)[0][0]
        plot_csi_data(
            magnitudes[room_idx],
            phases[room_idx],
            room
        )
    
    # Verify feature-label alignment
    logger.info("\nVerifying feature-label alignment:")
    logger.info(f"Number of samples: {len(rooms)}")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of unique rooms: {len(np.unique(rooms))}")
    logger.info(f"Number of zones: {len(np.unique(zones))}")
    
    # Check for any NaN or infinite values
    logger.info("\nChecking for invalid values:")
    logger.info(f"NaN values in features: {np.isnan(X).any()}")
    logger.info(f"Infinite values in features: {np.isinf(X).any()}")
    
    # Verify zone distribution
    logger.info("\nZone distribution verification:")
    for zone in range(5):
        zone_mask = zones == zone
        zone_rooms = rooms[zone_mask]
        logger.info(f"\nZone {zone}:")
        logger.info(f"  Samples: {np.sum(zone_mask)}")
        logger.info(f"  Unique rooms: {len(np.unique(zone_rooms))}")
        logger.info(f"  Rooms: {', '.join(sorted(np.unique(zone_rooms))[:5])}...")

if __name__ == "__main__":
    main()
