import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.dataset import load_and_merge_data, extract_sequences
from src.data_pipeline.thermodynamics import ThermodynamicsEngine
from src.models.pinn import DigitalTwinModel
from src.models.loss import PhysicsConstrainedLoss
from scipy.stats import wilcoxon

def run_loeo_cv_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = load_and_merge_data()
    thermo = ThermodynamicsEngine()
    
    # We will compute physics features for the whole dataset first
    phys_df = thermo.extract_physics_features(df)
    phys_features = phys_df.values
    
    raw_cols = ['Tamb', 'Pamb', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'RPM', 'Fuel_Flow', 'Altitude_m', 'Mach']
    raw_features = df[raw_cols].values
    
    # Combined features: raw + phys (drop Altitude and Mach from phys since they are in raw)
    phys_features_no_atm = phys_df.drop(columns=['Altitude_m', 'Mach'], errors='ignore').values
    combined_features = np.concatenate([raw_features, phys_features_no_atm], axis=1)
    
    target_cols = ['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 'OverallHealth', 'Thrust_N', 'TSFC_g_N_s']
    targets = df[target_cols].values
    
    engine_ids = df['EngineID'].unique()
    
    variants = [
        {"name": "MLP (N=3)", "seq_len": 3, "model_type": "mlp"},
        {"name": "GRU (N=3)", "seq_len": 3, "model_type": "gru"},
        {"name": "MLP (N=5)", "seq_len": 5, "model_type": "mlp"},
        {"name": "GRU (N=5)", "seq_len": 5, "model_type": "gru"}
    ]
    
    results = {v["name"]: {"rmse": [], "tsfc_rmse": []} for v in variants}
    
    print("\nStarting LOEO-CV (Leave-One-Engine-Out Cross Validation) Showdown...")
    print("Phase 3: GRU vs MLP Sequence Modeling Ablation\n")
    
    for held_out_engine in engine_ids:
        train_idx = df['EngineID'] != held_out_engine
        test_idx = df['EngineID'] == held_out_engine
        
        # Scaling
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
        
        for v in variants:
            seq_len = v["seq_len"]
            model_type = v["model_type"]
            
            # Extract sequences (with zero padding & masking for engines < N cycles)
            X_seq_train, y_seq_train, mask_train, ff_train = extract_sequences(df_train, X_combined_train_flat, y_train_flat, seq_length=seq_len)
            X_seq_test, y_seq_test, mask_test, ff_test = extract_sequences(df_test, X_combined_test_flat, y_test_flat, seq_length=seq_len)
            
            X_train_t = torch.tensor(X_seq_train, dtype=torch.float32).to(device)
            X_test_t = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_seq_train, dtype=torch.float32).to(device)
            y_test_t = torch.tensor(y_seq_test, dtype=torch.float32).to(device)
            
            # fuel flow in grams for TSFC calculation
            ff_train_t = torch.tensor(ff_train * 1000.0, dtype=torch.float32).to(device)
            ff_test_t = torch.tensor(ff_test * 1000.0, dtype=torch.float32).to(device)
            
            # The input feature dim is the last dimension of the combined features array
            input_dim = combined_features.shape[1] 
            if model_type == 'mlp':
                input_dim = input_dim * seq_len
                
            model = DigitalTwinModel(input_dim=input_dim, hidden_dim=32, dropout_rate=0.1, model_type=model_type).to(device)
            criterion = PhysicsConstrainedLoss(alpha=1.0, beta_health=1.0) 
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            
            # Simple fixed epochs for LOEO-CV speed
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
                thrust_pred_norm = preds_norm[4].squeeze()
                
                overall_health_pred = overall_health_pred_norm * target_scale[3] + target_mean[3]
                thrust_pred = thrust_pred_norm * target_scale[4] + target_mean[4]
                
                rmse = torch.sqrt(torch.mean((overall_health_pred - y_test_t[:, 3])**2)).item()
                
                # Deterministic TSFC Calculation
                tsfc_pred = ff_test_t / (torch.abs(thrust_pred) + 1e-6)
                tsfc_rmse = torch.sqrt(torch.mean((tsfc_pred - y_test_t[:, 5])**2)).item()
                
                results[v["name"]]["rmse"].append(rmse)
                results[v["name"]]["tsfc_rmse"].append(tsfc_rmse)
                
    print("\n==================================================")
    print("   LOEO-CV PHASE 3 BENCHMARK RESULTS")
    print("==================================================")
    
    final_results_for_json = []
    
    for v in variants:
        mean_rmse = np.mean(results[v["name"]]["rmse"])
        std_rmse = np.std(results[v["name"]]["rmse"])
        mean_tsfc = np.mean(results[v["name"]]["tsfc_rmse"])
        
        print(f"{v['name']}: Health RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}  |  TSFC RMSE = {mean_tsfc:.4f}")
        final_results_for_json.append({
            "name": v["name"],
            "health_rmse": mean_rmse,
            "tsfc_violation": mean_tsfc 
        })
        
    stat1, p_value1 = wilcoxon(results["MLP (N=3)"]["rmse"], results["GRU (N=3)"]["rmse"])
    stat2, p_value2 = wilcoxon(results["MLP (N=5)"]["rmse"], results["GRU (N=5)"]["rmse"])
    
    print("\n--- Statistical Significance ---")
    print(f"Wilcoxon paired test (MLP vs GRU for N=3) p-value: {p_value1:.4f}")
    print(f"Wilcoxon paired test (MLP vs GRU for N=5) p-value: {p_value2:.4f}")
    
    print("\n--- Surrogate Speed Benchmark ---")
    N_TRIALS = len(df)
    
    # Warm up Path A
    for i in range(5):
        _ = thermo.extract_physics_features(df.iloc[[i]])
        
    t0 = time.perf_counter()
    for i in range(N_TRIALS):
        _ = thermo.extract_physics_features(df.iloc[[i]])
    slow_path_s = time.perf_counter() - t0
    
    # Path B (Batched NN inference)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(X_test_t) # Using the last trained model
    fast_path_s = time.perf_counter() - t0
    
    speedup = slow_path_s / fast_path_s
    per_sample_us = (fast_path_s / N_TRIALS) * 1_000_000
    n_params = sum(p.numel() for p in model.parameters())
    
    print(f"Slow path (recompute): {slow_path_s:.6f} s total")
    print(f"Fast path (surrogate): {fast_path_s:.6f} s total")
    print(f"Speedup: {speedup:.1f}x | {per_sample_us:.3f} µs/sample | {n_params:,} params")
    
    os.makedirs("public/data", exist_ok=True)
    import json
    with open("public/data/benchmark_results.json", "w") as f:
        json.dump({
            "models": final_results_for_json, # Matched dashboard expectation
            "surrogate": {
                "slow_path_s": slow_path_s,
                "fast_path_s": fast_path_s,
                "surrogate_ms": per_sample_us / 1000.0,
                "speedup_x": speedup,
                "n_params": n_params
            }
        }, f, indent=2)

if __name__ == "__main__":
    run_loeo_cv_benchmark()
