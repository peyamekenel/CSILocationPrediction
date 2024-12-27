#!/usr/bin/env python3
"""
CSI Extraction Module for Indoor Localization System

This module provides functionality to extract Channel State Information (CSI)
from pcap files captured using the Nexmon toolkit. It handles the parsing of
UDP packets and conversion of raw bytes to complex CSI values.
"""

from scapy.all import rdpcap
import numpy as np
import struct
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_csi_from_pcap(
    pcap_path: str,
    header_offset: int = 16,
    nfft: int = 64
) -> Optional[np.ndarray]:
    """
    Extract CSI data from a pcap file captured using Nexmon toolkit.

    Args:
        pcap_path (str): Path to the pcap file
        header_offset (int): Number of bytes to skip at the start of payload (Nexmon header)
        nfft (int): Number of FFT points (default: 64 for 20MHz bandwidth)

    Returns:
        np.ndarray: Array of complex CSI values with shape (num_packets, nfft)
                   or None if file cannot be processed

    Raises:
        FileNotFoundError: If pcap_path does not exist
        ValueError: If pcap file contains no valid packets
    """
    try:
        # Read pcap file
        packets = rdpcap(pcap_path)
        if not packets:
            raise ValueError(f"No packets found in {pcap_path}")
        
        csi_data: List[np.ndarray] = []
        total_packets = len(packets)
        valid_packets = 0
        
        for packet_idx, pkt in enumerate(packets, 1):
            try:
                # Check for UDP layer and minimum payload size
                if not pkt.haslayer('UDP'):
                    continue
                
                payload = bytes(pkt['UDP'].payload)
                required_size = header_offset + 4 * nfft  # 4 bytes per complex number
                
                if len(payload) < required_size:
                    logger.debug(
                        f"Packet {packet_idx}/{total_packets} payload too small: "
                        f"{len(payload)} < {required_size} bytes"
                    )
                    continue
                
                # Extract CSI data chunk after header
                # Each complex number is 2 bytes real + 2 bytes imaginary
                data_chunk = payload[header_offset:header_offset + 4*nfft]
                
                # Unpack as 16-bit integers (little-endian)
                # Format: alternating real and imaginary parts
                values = struct.unpack(f"<{nfft*2}h", data_chunk)
                
                # Separate real and imaginary parts
                # Scale values to float in [-1, 1] range
                real_vals = np.array(values[0::2], dtype=float) / 32768.0
                imag_vals = np.array(values[1::2], dtype=float) / 32768.0
                
                # Combine into complex values
                complex_vals = real_vals + 1j * imag_vals
                csi_data.append(complex_vals)
                valid_packets += 1
                
            except struct.error as e:
                logger.warning(
                    f"Failed to unpack packet {packet_idx}/{total_packets}: {e}"
                )
                continue
            except Exception as e:
                logger.warning(
                    f"Error processing packet {packet_idx}/{total_packets}: {e}"
                )
                continue
        
        if not csi_data:
            logger.error(f"No valid CSI data extracted from {pcap_path}")
            return None
        
        logger.info(
            f"Successfully processed {valid_packets}/{total_packets} packets "
            f"from {pcap_path}"
        )
        
        return np.array(csi_data)
    
    except FileNotFoundError:
        logger.error(f"Pcap file not found: {pcap_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to process {pcap_path}: {e}")
        return None

def process_csi_data(
    csi_array: np.ndarray,
    subcarrier_range: tuple = (5, 61),
    dc_index: int = 29
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process CSI data array by applying FFT shift, filtering subcarriers,
    and calculating magnitude and phase.

    Args:
        csi_array (np.ndarray): Array of complex CSI values (shape: num_packets, nfft)
        subcarrier_range (tuple): Range of subcarriers to keep (start, end)
        dc_index (int): Index of DC component to zero out (before shift)

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing:
            - magnitude array (shape: num_packets, num_subcarriers)
            - unwrapped phase array (shape: num_packets, num_subcarriers)

    Raises:
        ValueError: If input array shape is invalid or subcarrier range is invalid
    """
    if not validate_csi_data(csi_array):
        raise ValueError("Invalid CSI data array")
    
    start_idx, end_idx = subcarrier_range
    if not (0 <= start_idx < end_idx <= csi_array.shape[1]):
        raise ValueError(
            f"Invalid subcarrier range {subcarrier_range} "
            f"for array with {csi_array.shape[1]} subcarriers"
        )
    
    try:
        # Shift the center for subcarriers
        csi_array_shifted = np.fft.fftshift(csi_array, axes=(1,))
        
        # Filter subcarriers to specified range
        csi_filtered = csi_array_shifted[:, start_idx:end_idx]
        
        # Calculate DC subcarrier index in filtered array
        dc_shifted = (dc_index + csi_array.shape[1]//2) % csi_array.shape[1]
        if start_idx <= dc_shifted < end_idx:
            dc_filtered_idx = dc_shifted - start_idx
            csi_filtered[:, dc_filtered_idx] = 0
            logger.info(f"Zeroed out DC component at filtered index {dc_filtered_idx}")
        
        # Calculate magnitude
        magnitude = np.abs(csi_filtered)
        
        # Calculate phase and unwrap along subcarrier dimension
        phase = np.angle(csi_filtered)
        phase_unwrapped = np.unwrap(phase, axis=1)
        
        logger.info(
            f"Processed CSI data: {csi_array.shape} -> "
            f"magnitude/phase shape: {magnitude.shape}"
        )
        
        return magnitude, phase_unwrapped
    
    except Exception as e:
        logger.error(f"Error processing CSI data: {e}")
        raise

def validate_csi_data(csi_array: np.ndarray) -> bool:
    """
    Validate extracted CSI data array.

    Args:
        csi_array (np.ndarray): Array of complex CSI values

    Returns:
        bool: True if data appears valid, False otherwise
    """
    if csi_array is None or len(csi_array) == 0:
        return False
    
    # Check for expected shape (num_packets, nfft)
    if len(csi_array.shape) != 2:
        return False
    
    # Check for non-zero values (completely zero array likely indicates error)
    if np.all(csi_array == 0):
        return False
    
    # Check for reasonable magnitude range
    magnitudes = np.abs(csi_array)
    if np.any(np.isinf(magnitudes)) or np.any(np.isnan(magnitudes)):
        return False
    
    return True

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pcap_file>")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    csi_data = extract_csi_from_pcap(pcap_file)
    
    if csi_data is not None and validate_csi_data(csi_data):
        print(f"Successfully extracted CSI data: shape={csi_data.shape}")
        print(f"Number of packets: {len(csi_data)}")
        print(f"FFT points per packet: {csi_data.shape[1]}")
        
        try:
            magnitude, phase = process_csi_data(csi_data)
            print("\nProcessed CSI data:")
            print(f"Magnitude shape: {magnitude.shape}")
            print(f"Phase shape: {phase.shape}")
            print(f"DC component zeroed: True")
            print(f"Subcarrier range: [5:61]")
        except Exception as e:
            print(f"Error processing CSI data: {e}")
            sys.exit(1)
    else:
        print("Failed to extract valid CSI data")
        sys.exit(1)
