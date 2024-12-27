import pandas as pd
import numpy as np
from scapy.all import rdpcap
import os

def load_location_data():
    """Load and examine location information from Excel files."""
    coords_df = pd.read_excel('locations/location_coords.xlsx')
    info_df = pd.read_excel('locations/location_info.xlsx')
    
    print("\nLocation Coordinates Dataset Info:")
    print(coords_df.info())
    print("\nSample coordinates:")
    print(coords_df.head())
    
    print("\nLocation Info Dataset Info:")
    print(info_df.info())
    print("\nSample location info:")
    print(info_df.head())
    
    return coords_df, info_df

def examine_pcap_sample():
    """Examine a sample pcap file to understand CSI/RSSI data structure."""
    # Get first pcap file from ref directory
    ref_files = [f for f in os.listdir('data/ref') if f.endswith('.pcap')]
    if not ref_files:
        print("No pcap files found in data/ref directory")
        return
        
    sample_file = os.path.join('data/ref', ref_files[0])
    print(f"\nExamining sample pcap file: {sample_file}")
    
    # Read pcap file
    packets = rdpcap(sample_file)
    print(f"Number of packets in sample file: {len(packets)}")
    
    # Examine first packet
    if len(packets) > 0:
        print("\nSample packet structure:")
        print(packets[0].show())

def main():
    print("=== Dataset Exploration ===")
    
    # Load and examine location data
    coords_df, info_df = load_location_data()
    
    # Examine pcap sample
    examine_pcap_sample()

if __name__ == "__main__":
    main()
