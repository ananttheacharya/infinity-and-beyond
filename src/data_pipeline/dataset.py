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

def prepare_dataloaders(df, train_idx, val_idx, batch_size=64, use_physics_features=True):
    """
    Prepares DataLoaders for train and validation splits.
    Scales features based ONLY on the train split.
    """
    thermo_engine = ThermodynamicsEngine()
    
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    
    if use_physics_features:
        # Extract physics features
        X_train_df = thermo_engine.extract_physics_features(df_train)
        X_val_df = thermo_engine.extract_physics_features(df_val)
    else:
        # Raw features
        sensor_cols = ['Tamb', 'Pamb', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'RPM', 'Fuel_Flow', 'Altitude_m', 'Mach']
        X_train_df = df_train[sensor_cols]
        X_val_df = df_val[sensor_cols]
        
    X_train_raw = X_train_df.values
    X_val_raw = X_val_df.values
    
    # Fit scaler on train ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    # Extract targets
    target_cols = ['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 'OverallHealth', 'Thrust_N', 'TSFC_g_N_s']
    y_train_raw = df_train[target_cols].values
    y_val_raw = df_val[target_cols].values
    
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train_raw)
    y_val = target_scaler.transform(y_val_raw)
    
    fuel_flow_train = df_train['Fuel_Flow'].values
    fuel_flow_val = df_val['Fuel_Flow'].values
    
    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    ff_train_t = torch.tensor(fuel_flow_train, dtype=torch.float32)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    ff_val_t = torch.tensor(fuel_flow_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t, ff_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t, ff_val_t)
    
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
