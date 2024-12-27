import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from csi_extraction import extract_csi_from_pcap, process_csi_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_room_info(filename: str) -> Tuple[str, int]:
    """
    Extract room number and space ID from filename.
    Examples: 
        'ref_204_1_3500.pcap' -> ('204', 1)
        'ref_A223_1_3500.pcap' -> ('A223', 1)
    
    Args:
        filename (str): Name of the pcap file
        
    Returns:
        Tuple[str, int]: (room_number, space_id)
        
    Raises:
        ValueError: If filename format is invalid
    """
    try:
        parts = filename.split('_')
        if len(parts) < 4:
            raise ValueError(f"Invalid filename format: {filename}")
        
        room_number = parts[1]  # Keep as string to preserve alphanumeric values
        space_id = int(parts[2])
        return room_number, space_id
    except (IndexError, ValueError) as e:
        logger.error(f"Failed to extract room info from {filename}: {str(e)}")
        raise ValueError(f"Invalid filename format: {filename}") from e

def process_pcap_directory(
    pcap_dir: str,
    output_dir: str,
    prefix: str = ""
) -> Dict[str, np.ndarray]:
    """
    Process all pcap files in a directory and save extracted features.
    
    Args:
        pcap_dir (str): Directory containing pcap files
        output_dir (str): Directory to save processed features
        prefix (str): Prefix for output files (e.g., 'train' or 'test')
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'magnitudes': Array of shape (n_samples, n_packets, n_subcarriers)
            - 'phases': Array of shape (n_samples, n_packets, n_subcarriers)
            - 'room_numbers': Array of room numbers
            - 'space_ids': Array of space IDs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to store processed data
    magnitudes_list = []
    phases_list = []
    room_numbers = []
    space_ids = []
    
    # Process each pcap file
    pcap_files = [f for f in os.listdir(pcap_dir) if f.endswith('.pcap')]
    total_files = len(pcap_files)
    max_packets = 3500  # Maximum number of packets to store per file
    
    for idx, filename in enumerate(pcap_files, 1):
        logger.info(f"Processing file {idx}/{total_files}: {filename}")
        
        try:
            # Skip files that don't match the expected format
            if not (filename.startswith('ref_') or filename.startswith('test_')):
                logger.warning(f"Skipping file with non-standard name: {filename}")
                continue
            
            # Extract room information
            room_number, space_id = extract_room_info(filename)
            
            # Process pcap file
            pcap_path = os.path.join(pcap_dir, filename)
            csi_data = extract_csi_from_pcap(pcap_path)
            
            if csi_data is not None:
                # Process CSI data
                magnitude, phase = process_csi_data(csi_data)
                
                # Pad or truncate to max_packets
                if magnitude.shape[0] < max_packets:
                    # Pad with zeros
                    pad_size = max_packets - magnitude.shape[0]
                    magnitude = np.pad(
                        magnitude,
                        ((0, pad_size), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
                    phase = np.pad(
                        phase,
                        ((0, pad_size), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
                elif magnitude.shape[0] > max_packets:
                    # Truncate
                    magnitude = magnitude[:max_packets]
                    phase = phase[:max_packets]
                
                # Store results
                magnitudes_list.append(magnitude)
                phases_list.append(phase)
                room_numbers.append(room_number)
                space_ids.append(space_id)
                
                logger.info(
                    f"Processed and standardized to {magnitude.shape[0]} packets "
                    f"with {magnitude.shape[1]} subcarriers"
                )
            else:
                logger.warning(f"Failed to extract CSI data from {filename}")
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    # Convert lists to arrays
    features = {
        'magnitudes': np.array(magnitudes_list),
        'phases': np.array(phases_list),
        'room_numbers': np.array(room_numbers, dtype=object),  # Use object dtype for strings
        'space_ids': np.array(space_ids)
    }
    
    # Save processed features
    output_path = os.path.join(output_dir, f"{prefix}_features.npz")
    np.savez(
        output_path,
        magnitudes=features['magnitudes'],
        phases=features['phases'],
        room_numbers=features['room_numbers'],
        space_ids=features['space_ids']
    )
    
    logger.info(f"Saved processed features to {output_path}")
    return features

if __name__ == "__main__":
    # Process reference and test data
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "processed_features"
    
    logger.info("Processing reference data...")
    ref_features = process_pcap_directory(
        str(data_dir / "ref"),
        str(output_dir),
        "reference"
    )
    
    logger.info("Processing test data...")
    test_features = process_pcap_directory(
        str(data_dir / "test"),
        str(output_dir),
        "test"
    )
    
    logger.info("Preprocessing complete!")
    logger.info(f"Reference samples: {len(ref_features['room_numbers'])}")
    logger.info(f"Test samples: {len(test_features['room_numbers'])}")
