import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.thermodynamics import ThermodynamicsEngine
from src.models.pinn import DigitalTwinModel
from src.models.loss import PhysicsConstrainedLoss
from src.evaluation.metrics import compute_tsfc_violation

def run_loeo_cv():
    print("Starting LOEO-CV (Leave-One-Engine-Out Cross Validation) Debug Harness")
    df = pd.read_csv('Dataset/turbojet_complete_dataset.csv')
    df.rename(columns={
        'Tamb_K': 'Tamb', 'Pamb_Pa': 'Pamb', 'T2_K': 'T2', 'P2_Pa': 'P2',
        'T3_K': 'T3', 'P3_Pa': 'P3', 'T4_K': 'T4', 'P4_Pa': 'P4',
        'RPM_rev_min': 'RPM', 'FuelFlow_kg_s': 'Fuel_Flow'
    }, inplace=True)
    
    thermo = ThermodynamicsEngine()
    phys_df = thermo.extract_physics_features(df)
    
    # Feature inputs
    phys_features = phys_df.values
    raw_cols = ['Tamb', 'Pamb', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'RPM', 'Fuel_Flow', 'Altitude_m', 'Mach']
    raw_features = df[raw_cols].values
    
    # Target values (6 heads)
    targets = df[['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 'OverallHealth', 'Thrust_N', 'TSFC_g_N_s']].values
    
    engine_ids = df['EngineID'].unique()
    
    raw_rmse_scores = []
    phys_rmse_scores = []
    
    for held_out_engine in engine_ids:
        # Split logic
        train_idx = df['EngineID'] != held_out_engine
        test_idx = df['EngineID'] == held_out_engine
        
        # Scaling
        raw_scaler = StandardScaler().fit(raw_features[train_idx])
        phys_scaler = StandardScaler().fit(phys_features[train_idx])
        target_scaler = StandardScaler().fit(targets[train_idx])
        
        # Datasets
        X_raw_train = torch.tensor(raw_scaler.transform(raw_features[train_idx]), dtype=torch.float32)
        X_raw_test = torch.tensor(raw_scaler.transform(raw_features[test_idx]), dtype=torch.float32)
        
        X_phys_train = torch.tensor(phys_scaler.transform(phys_features[train_idx]), dtype=torch.float32)
        X_phys_test = torch.tensor(phys_scaler.transform(phys_features[test_idx]), dtype=torch.float32)
        
        y_train = torch.tensor(target_scaler.transform(targets[train_idx]), dtype=torch.float32)
        y_test = torch.tensor(targets[test_idx], dtype=torch.float32) # Keep unscaled for RMSE calculation later
        
        target_mean = torch.tensor(target_scaler.mean_, dtype=torch.float32)
        target_scale = torch.tensor(target_scaler.scale_, dtype=torch.float32)
        
        # Model Training - Raw
        model_raw = DigitalTwinModel(input_dim=raw_features.shape[1], hidden_dim=32, dropout_rate=0.1)
        optimizer_raw = torch.optim.Adam(model_raw.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(150):
            model_raw.train()
            optimizer_raw.zero_grad()
            preds = model_raw(X_raw_train)
            preds = torch.cat(preds, dim=1)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer_raw.step()
            
        # Model Training - PhysFeat
        model_phys = DigitalTwinModel(input_dim=phys_features.shape[1], hidden_dim=32, dropout_rate=0.1)
        optimizer_phys = torch.optim.Adam(model_phys.parameters(), lr=0.001)
        
        for epoch in range(150):
            model_phys.train()
            optimizer_phys.zero_grad()
            preds = model_phys(X_phys_train)
            preds = torch.cat(preds, dim=1)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer_phys.step()
            
        # Evaluation
        model_raw.eval()
        model_phys.eval()
        with torch.no_grad():
            preds_raw_norm = torch.cat(model_raw(X_raw_test), dim=1)
            preds_phys_norm = torch.cat(model_phys(X_phys_test), dim=1)
            
            # Denormalize
            preds_raw = preds_raw_norm * target_scale + target_mean
            preds_phys = preds_phys_norm * target_scale + target_mean
            
            # Overall Health is index 3
            rmse_raw = torch.sqrt(torch.mean((preds_raw[:, 3] - y_test[:, 3])**2)).item()
            rmse_phys = torch.sqrt(torch.mean((preds_phys[:, 3] - y_test[:, 3])**2)).item()
            
            raw_rmse_scores.append(rmse_raw)
            phys_rmse_scores.append(rmse_phys)
            
        print(f"Fold Engine {held_out_engine} | Raw RMSE: {rmse_raw:.4f} | PhysFeat RMSE: {rmse_phys:.4f}")

    print("\n--- Final LOEO-CV Results ---")
    print(f"Baseline-Raw RMSE:       {np.mean(raw_rmse_scores):.4f} ± {np.std(raw_rmse_scores):.4f}")
    print(f"Baseline-PhysFeat RMSE:  {np.mean(phys_rmse_scores):.4f} ± {np.std(phys_rmse_scores):.4f}")
    
    # Wilcoxon Signed-Rank Test
    stat, p_value = wilcoxon(raw_rmse_scores, phys_rmse_scores)
    print(f"\nWilcoxon paired test p-value: {p_value:.4f}")
    
    # Effect Size (matched-pairs rank-biserial correlation)
    diffs = np.array(phys_rmse_scores) - np.array(raw_rmse_scores)
    w_plus = np.sum(np.abs(diffs[diffs > 0]))
    w_minus = np.sum(np.abs(diffs[diffs < 0]))
    effect_size = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0
    print(f"Effect Size (Rank-Biserial): {effect_size:.4f} (Negative means PhysFeat is better)")

if __name__ == '__main__':
    run_loeo_cv()
