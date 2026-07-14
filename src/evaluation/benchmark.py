import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.dataset import load_and_merge_data, get_engine_split
from src.data_pipeline.thermodynamics import ThermodynamicsEngine
from src.models.pinn import DigitalTwinModel

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df_merged = load_and_merge_data()
    _, df_test = get_engine_split(df_merged, test_engines=[9, 10])
    
    thermo_engine = ThermodynamicsEngine()
    
    variants = [
        {"name": "Baseline-Raw", "use_physics": False},
        {"name": "Baseline-PhysFeat", "use_physics": True},
        {"name": "Full Model", "use_physics": True}
    ]
    
    print("\n==================================================")
    print("   ABLATION BENCHMARK RESULTS")
    print("==================================================")
    
    results = []
    
    for v in variants:
        model_name_lower = v["name"].replace(" ", "_").lower()
        model_path = f'dist/models/{model_name_lower}.pth'
        scaler_path = f'dist/models/{model_name_lower}_scaler.joblib'
        target_scaler_path = f'dist/models/{model_name_lower}_target_scaler.joblib'
        
        if not os.path.exists(model_path):
            print(f"Skipping {v['name']} - Model not found at {model_path}")
            continue
            
        scaler = joblib.load(scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        target_scale = torch.tensor(target_scaler.scale_, dtype=torch.float32).to(device)
        target_mean = torch.tensor(target_scaler.mean_, dtype=torch.float32).to(device)
        
        if v["use_physics"]:
            X_test_df = thermo_engine.extract_physics_features(df_test)
        else:
            sensor_cols = ['Tamb', 'Pamb', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'RPM', 'Fuel_Flow', 'Altitude_m', 'Mach']
            X_test_df = df_test[sensor_cols]
            
        X_test_scaled = scaler.transform(X_test_df.values)
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        
        target_cols = ['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 'OverallHealth', 'Thrust_N', 'TSFC_g_N_s']
        y_test_tensor = torch.tensor(df_test[target_cols].values, dtype=torch.float32).to(device)
        fuel_flow_g = torch.tensor(df_test['Fuel_Flow'].values * 1000.0, dtype=torch.float32).to(device)
        
        model = DigitalTwinModel(input_dim=X_tensor.shape[1], hidden_dim=32, dropout_rate=0.1).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        # Predictive Performance and TSFC Violation
        with torch.no_grad():
            preds = model(X_tensor)
            overall_health_pred_norm = preds[3].squeeze()
            thrust_pred_norm = preds[4].squeeze()
            
            # Denormalize
            overall_health_pred = overall_health_pred_norm * target_scale[3] + target_mean[3]
            thrust_pred = thrust_pred_norm * target_scale[4] + target_mean[4]
            
            rmse = torch.sqrt(torch.mean((overall_health_pred - y_test_tensor[:, 3])**2))
            
            # Deterministic TSFC Calculation
            tsfc_pred = fuel_flow_g / (torch.abs(thrust_pred) + 1e-6)
            tsfc_rmse = torch.sqrt(torch.mean((tsfc_pred - y_test_tensor[:, 5])**2))
            
            print(f"{v['name']}: TSFC RMSE={tsfc_rmse.item():.4f}  Overall-health RMSE={rmse.item():.4f}")
            results.append({"name": v["name"], "rmse": rmse.item(), "tsfc_rmse": tsfc_rmse.item()})
            
        # Calibration (MC-Dropout) for Full Model
        if v["name"] == "Full Model":
            print("\n--- Calibration (MC-Dropout Coverage) ---")
            stats = model.predict_with_uncertainty(X_tensor, num_samples=50)
            overall_mean_norm, overall_std_norm = stats[3]
            overall_mean_norm = overall_mean_norm.squeeze()
            overall_std_norm = overall_std_norm.squeeze()
            
            # Denormalize
            overall_mean = overall_mean_norm * target_scale[3] + target_mean[3]
            overall_std = overall_std_norm * target_scale[3]
            
            # Print sample to diagnose
            print("Sample raw values (True, Mean, Std):")
            for i in range(min(5, len(y_test_tensor))):
                print(f"  {y_test_tensor[i, 3].item():.4f} | {overall_mean[i].item():.4f} | {overall_std[i].item():.4f}")
            
            # Check how often true value is within mean +/- 1 std
            lower_bound = overall_mean - overall_std
            upper_bound = overall_mean + overall_std
            true_overall = y_test_tensor[:, 3]
            
            within_bounds = ((true_overall >= lower_bound) & (true_overall <= upper_bound)).float()
            coverage_pct = within_bounds.mean().item() * 100
            print(f"Coverage (mean ± 1 std): {coverage_pct:.1f}%")
            
            print("\n--- Surrogate Speed Benchmark ---")
            N_TRIALS = len(df_test)
            
            # Warm up Path A
            for i in range(5):
                _ = thermo_engine.extract_physics_features(df_test.iloc[[i]])
                
            # Path A (Iterative physics extraction)
            t0 = time.perf_counter()
            for i in range(N_TRIALS):
                _ = thermo_engine.extract_physics_features(df_test.iloc[[i]])
            slow_path_s = time.perf_counter() - t0
            
            # Warm up Path B
            with torch.no_grad():
                for _ in range(5):
                    _ = model(X_tensor[:1])
            
            # Path B (Batched NN inference)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(X_tensor)
            fast_path_s = time.perf_counter() - t0
            
            speedup = slow_path_s / fast_path_s
            per_sample_us = (fast_path_s / N_TRIALS) * 1_000_000
            n_params = sum(p.numel() for p in model.parameters())
            
            print(f"Slow path (recompute): {slow_path_s:.6f} s total")
            print(f"Fast path (surrogate): {fast_path_s:.6f} s total")
            print(f"Speedup: {speedup:.1f}x | {per_sample_us:.3f} µs/sample | {n_params:,} params")
            
            # Save results to JSON for dashboard
            import json
            os.makedirs("public/data", exist_ok=True)
            with open("public/data/benchmark_results.json", "w") as f:
                json.dump({
                    "variants": results,
                    "surrogate": {
                        "slow_path_s": slow_path_s,
                        "fast_path_s": fast_path_s,
                        "surrogate_ms": per_sample_us / 1000.0,
                        "speedup_x": speedup,
                        "n_params": n_params
                    },
                    "calibration": {
                        "coverage_1std_pct": coverage_pct,
                        "n_held_out": len(df_test),
                        "description": "% of true values falling within predicted mean ± 1 std on held-out test engines"
                    }
                }, f, indent=2)

if __name__ == "__main__":
    benchmark()
