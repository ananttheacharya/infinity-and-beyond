import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.dataset import load_and_merge_data, extract_sequences
from src.data_pipeline.thermodynamics import ThermodynamicsEngine
from src.models.transfer import TransferredDigitalTwinModel
from src.models.loss import PhysicsConstrainedLoss
from scipy.stats import wilcoxon

def get_model(device, pretrained_path=None, freeze_encoder=True):
    model = TransferredDigitalTwinModel().to(device)
    
    if pretrained_path and os.path.exists(pretrained_path):
        model.encoder.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
        
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
            
    return model

def run_linear_probe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = load_and_merge_data()
    thermo = ThermodynamicsEngine()
    
    # Precompute physics features
    phys_df = thermo.extract_physics_features(df)
    phys_features = phys_df.values
    raw_cols = ['Tamb', 'Pamb', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'RPM', 'Fuel_Flow', 'Altitude_m', 'Mach']
    raw_features = df[raw_cols].values
    phys_features_no_atm = phys_df.drop(columns=['Altitude_m', 'Mach'], errors='ignore').values
    combined_features = np.concatenate([raw_features, phys_features_no_atm], axis=1)
    
    target_cols = ['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 'OverallHealth', 'Thrust_N', 'TSFC_g_N_s']
    targets = df[target_cols].values
    engine_ids = df['EngineID'].unique()
    
    variants = [
        {"name": "Frozen Random Encoder", "pretrained": None},
        {"name": "Frozen Pretrained Encoder", "pretrained": "models/ncmapss_pretrained_encoder.pth"}
    ]
    
    results = {v["name"]: {"rmse": []} for v in variants}
    seq_len = 5
    
    print("\nStarting Phase 1 Linear Probe (10-Seed Ablation equivalent via LOEO-CV)...")
    
    for held_out_engine in engine_ids:
        train_idx = df['EngineID'] != held_out_engine
        test_idx = df['EngineID'] == held_out_engine
        
        from sklearn.preprocessing import StandardScaler
        combined_scaler = StandardScaler().fit(combined_features[train_idx])
        target_scaler = StandardScaler().fit(targets[train_idx])
        
        X_combined_train_flat = combined_scaler.transform(combined_features[train_idx])
        X_combined_test_flat = combined_scaler.transform(combined_features[test_idx])
        
        y_train_flat = target_scaler.transform(targets[train_idx])
        y_test_flat = targets[test_idx]
        
        df_train = df[train_idx].copy()
        df_test = df[test_idx].copy()
        
        target_mean = torch.tensor(target_scaler.mean_, dtype=torch.float32).to(device)
        target_scale = torch.tensor(target_scaler.scale_, dtype=torch.float32).to(device)
        
        X_seq_train, y_seq_train, mask_train, ff_train = extract_sequences(df_train, X_combined_train_flat, y_train_flat, seq_length=seq_len)
        X_seq_test, y_seq_test, mask_test, ff_test = extract_sequences(df_test, X_combined_test_flat, y_test_flat, seq_length=seq_len)
        
        X_train_t = torch.tensor(X_seq_train, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_seq_train, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_seq_test, dtype=torch.float32).to(device)
        ff_train_t = torch.tensor(ff_train * 1000.0, dtype=torch.float32).to(device)
        
        for v in variants:
            model = get_model(device, pretrained_path=v["pretrained"], freeze_encoder=True)
            criterion = PhysicsConstrainedLoss(alpha=1.0, beta_health=1.0) 
            # Train only adapter and heads
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)
            
            for epoch in range(120):
                model.train()
                optimizer.zero_grad()
                preds = model(X_train_t)
                total_loss, mse_total, _, _ = criterion(preds, y_train_t, ff_train_t, target_mean, target_scale)
                total_loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                preds_norm = model(X_test_t)
                overall_health_pred_norm = preds_norm[3].squeeze()
                overall_health_pred = overall_health_pred_norm * target_scale[3] + target_mean[3]
                rmse = torch.sqrt(torch.mean((overall_health_pred - y_test_t[:, 3])**2)).item()
                results[v["name"]]["rmse"].append(rmse)
                
    print("\n==================================================")
    print("   LINEAR PROBE ABLATION RESULTS (Overall Health)")
    print("==================================================")
    
    for v in variants:
        mean_rmse = np.mean(results[v["name"]]["rmse"])
        std_rmse = np.std(results[v["name"]]["rmse"])
        print(f"{v['name']}: Health RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
        
    stat, p_value = wilcoxon(results["Frozen Random Encoder"]["rmse"], results["Frozen Pretrained Encoder"]["rmse"])
    print("\n--- Statistical Significance ---")
    print(f"Wilcoxon paired test p-value: {p_value:.4f}")
    
    if p_value < 0.05 and np.mean(results["Frozen Pretrained Encoder"]["rmse"]) < np.mean(results["Frozen Random Encoder"]["rmse"]):
        print("\nSUCCESS: Pretrained encoder performs significantly better than random!")
        print("Abort Criterion Avoided. We can proceed to full fine-tuning.")
    else:
        print("\nFAILURE: Pretrained encoder performs no better than random initialization.")
        print("ABORT CRITERION MET: Transfer learning did not learn useful turbojet representations.")

if __name__ == "__main__":
    run_linear_probe()
