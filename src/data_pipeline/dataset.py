import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import os
import sys

# Append root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.thermodynamics import ThermodynamicsEngine

def load_and_merge_data(dataset_dir='Dataset'):
    """
    Loads train, test, and ground truth datasets and merges them into a single dataframe.
    """
    train_path = os.path.join(dataset_dir, 'train.csv')
    test_path = os.path.join(dataset_dir, 'test.csv')
    gt_path = os.path.join(dataset_dir, 'ground_truth.csv')
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_gt = pd.read_csv(gt_path)
    
    # Merge train and test since they have the same structure and we found EngineID overlap
    df_all_sensors = pd.concat([df_train, df_test], ignore_index=True).drop_duplicates(subset=['EngineID', 'Cycle'])
    
    # Merge with ground truth
    df_merged = pd.merge(df_all_sensors, df_gt, on=['EngineID', 'Cycle'], how='inner')
    
    # Rename columns to match ThermodynamicsEngine expectations
    df_merged.rename(columns={
        'Tamb_K': 'Tamb', 'Pamb_Pa': 'Pamb', 'T2_K': 'T2', 'P2_Pa': 'P2',
        'T3_K': 'T3', 'P3_Pa': 'P3', 'T4_K': 'T4', 'P4_Pa': 'P4',
        'RPM_rev_min': 'RPM', 'FuelFlow_kg_s': 'Fuel_Flow'
    }, inplace=True)
    
    return df_merged

def extract_sequences(df, features_array, targets_array=None, seq_length=5):
    """
    Groups data by EngineID and creates sliding windows of length seq_length.
    Applies explicit zero-padding for engines with fewer cycles than seq_length.
    Returns (X_seq, y_seq, masks, fuel_flows)
    """
    X_seq, y_seq, masks, fuel_flows = [], [], [], []
    
    df_reset = df.reset_index(drop=True)
    grouped = df_reset.groupby('EngineID', sort=False)
    
    ff_array = df_reset['Fuel_Flow'].values
    
    for engine_id, group in grouped:
        indices = group.index.values
        X_group = features_array[indices]
        y_group = targets_array[indices] if targets_array is not None else None
        ff_group = ff_array[indices]
        
        num_cycles = len(indices)
        
        if num_cycles < seq_length:
            # Explicit zero padding
            pad_len = seq_length - num_cycles
            X_pad = np.zeros((pad_len, X_group.shape[1]), dtype=np.float32)
            X_padded = np.vstack([X_pad, X_group])
            mask = np.concatenate([np.zeros(pad_len), np.ones(num_cycles)])
            
            X_seq.append(X_padded)
            masks.append(mask)
            if y_group is not None:
                y_seq.append(y_group[-1]) # target is the last cycle
            fuel_flows.append(ff_group[-1])
        else:
            # Sliding window
            for i in range(num_cycles - seq_length + 1):
                X_seq.append(X_group[i:i+seq_length])
                masks.append(np.ones(seq_length))
                if y_group is not None:
                    y_seq.append(y_group[i+seq_length-1])
                fuel_flows.append(ff_group[i+seq_length-1])
                
    X_seq = np.array(X_seq, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32) if targets_array is not None else None
    fuel_flows = np.array(fuel_flows, dtype=np.float32)
    
    return X_seq, y_seq, masks, fuel_flows

def prepare_dataloaders(df, train_idx, val_idx, batch_size=64, use_physics_features=True, seq_length=5):
    """
    Prepares DataLoaders for train and validation splits.
    Scales features based ONLY on the train split.
    Now uses Combined Features by default and extracts temporal sequences.
    """
    thermo_engine = ThermodynamicsEngine()
    
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    
    sensor_cols = ['Tamb', 'Pamb', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'RPM', 'Fuel_Flow', 'Altitude_m', 'Mach']
    X_train_raw_df = df_train[sensor_cols]
    X_val_raw_df = df_val[sensor_cols]
    
    if use_physics_features:
        # Extract physics features
        X_train_phys = thermo_engine.extract_physics_features(df_train).drop(columns=['Altitude_m', 'Mach'], errors='ignore')
        X_val_phys = thermo_engine.extract_physics_features(df_val).drop(columns=['Altitude_m', 'Mach'], errors='ignore')
        
        # Combined features
        X_train_df = pd.concat([X_train_raw_df, X_train_phys], axis=1)
        X_val_df = pd.concat([X_val_raw_df, X_val_phys], axis=1)
    else:
        # Raw features only
        X_train_df = X_train_raw_df
        X_val_df = X_val_raw_df
        
    X_train_flat = X_train_df.values
    X_val_flat = X_val_df.values
    
    # Fit scaler on train ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    
    # Extract targets
    target_cols = ['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 'OverallHealth', 'Thrust_N', 'TSFC_g_N_s']
    y_train_raw = df_train[target_cols].values
    y_val_raw = df_val[target_cols].values
    
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw)
    y_val_scaled = target_scaler.transform(y_val_raw)
    
    # Extract Sequences
    if seq_length > 1:
        X_train_seq, y_train_seq, masks_train, ff_train = extract_sequences(df_train, X_train_scaled, y_train_scaled, seq_length)
        X_val_seq, y_val_seq, masks_val, ff_val = extract_sequences(df_val, X_val_scaled, y_val_scaled, seq_length)
    else:
        X_train_seq, y_train_seq, ff_train = X_train_scaled, y_train_scaled, df_train['Fuel_Flow'].values
        X_val_seq, y_val_seq, ff_val = X_val_scaled, y_val_scaled, df_val['Fuel_Flow'].values
        masks_train = np.ones((len(X_train_seq), 1))
        masks_val = np.ones((len(X_val_seq), 1))
    
    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
    ff_train_t = torch.tensor(ff_train, dtype=torch.float32)
    masks_train_t = torch.tensor(masks_train, dtype=torch.float32)
    
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.float32)
    ff_val_t = torch.tensor(ff_val, dtype=torch.float32)
    masks_val_t = torch.tensor(masks_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t, ff_train_t, masks_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t, ff_val_t, masks_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler, target_scaler, X_train_df.columns

def get_engine_split(df, test_engines=[9, 10]):
    """
    Splits the dataframe into a train/val pool and a held-out test pool based on EngineID.
    """
    test_mask = df['EngineID'].isin(test_engines)
    df_test = df[test_mask].reset_index(drop=True)
    df_train_val = df[~test_mask].reset_index(drop=True)
    
    return df_train_val, df_test
