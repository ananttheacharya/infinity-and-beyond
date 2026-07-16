import os
import h5py
import numpy as np
from tqdm import tqdm

def extract_ncmapss_pairs(h5_path, output_dir, seq_len=128):
    """
    Extracts positive pairs from N-CMAPSS for contrastive self-supervised learning.
    A positive pair is defined as two adjacent windows of length `seq_len` from the same flight.
    Since they are from the same flight, they have the exact same degradation stage, 
    but capture slightly different operating regimes (forcing the encoder to learn invariant health features).
    """
    print(f"Loading N-CMAPSS dataset from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        # Load arrays into memory
        A_dev = f['A_dev'][:]
        X_s_dev = f['X_s_dev'][:]
        
        # A_dev columns: ['unit', 'cycle', 'Fc', 'hs']
        units = A_dev[:, 0]
        cycles = A_dev[:, 1]
        
    print("Extracting positive pairs...")
    
    X1_list = []
    X2_list = []
    
    # We want blocks of size 2 * seq_len
    block_size = 2 * seq_len
    
    unique_units = np.unique(units)
    
    for u in tqdm(unique_units, desc="Processing Units"):
        unit_mask = (units == u)
        unit_cycles = cycles[unit_mask]
        unit_X_s = X_s_dev[unit_mask]
        
        unique_c = np.unique(unit_cycles)
        
        for c in unique_c:
            flight_mask = (unit_cycles == c)
            flight_data = unit_X_s[flight_mask]
            
            n_rows = len(flight_data)
            
            # Extract non-overlapping blocks
            for i in range(0, n_rows - block_size + 1, block_size):
                block = flight_data[i : i + block_size]
                X1_list.append(block[0 : seq_len])
                X2_list.append(block[seq_len : block_size])

    X1_array = np.array(X1_list, dtype=np.float32)
    X2_array = np.array(X2_list, dtype=np.float32)
    
    print(f"Extracted {len(X1_array)} positive pairs.")
    print(f"X1 shape: {X1_array.shape}, X2 shape: {X2_array.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'ncmapss_pairs_{seq_len}.npz')
    np.savez(out_file, X1=X1_array, X2=X2_array)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    DATASET_PATH = "Dataset/N-CMAPSS_DS03-012.h5"
    OUTPUT_DIR = "Dataset/processed"
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Could not find {DATASET_PATH}")
    else:
        extract_ncmapss_pairs(DATASET_PATH, OUTPUT_DIR, seq_len=128)
