import time
import requests
import sys
import os
import torch
import pandas as pd
import joblib
import torch.nn as nn

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.thermodynamics import ThermodynamicsEngine
from src.models.pinn import DigitalTwinModel
from src.models.pinn import DigitalTwinModel

def stream_telemetry():
    print("Starting LIVE Physics-Informed Digital Twin Telemetry Streamer...")
    url = "http://localhost:3000/api/telemetry"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")
    
    thermo = ThermodynamicsEngine()
    
    # Determine input dims
    df_dummy = pd.DataFrame({'Tamb': [288], 'Pamb': [101325], 'T2': [350], 'P2': [150000], 'T3': [900], 'P3': [900000], 'T4': [700], 'P4': [400000], 'RPM': [9000], 'Fuel_Flow': [1.0], 'Altitude_m': [0], 'Mach': [0]})
    phys_input_dim = thermo.extract_physics_features(df_dummy).shape[1]
    raw_input_dim = 12 # sensor cols
    
    # 1. Load the Models
    try:
        # Full Model
        full_model = DigitalTwinModel(input_dim=phys_input_dim, hidden_dim=32, dropout_rate=0.1).to(device)
        full_model.load_state_dict(torch.load('dist/models/full_model.pth', map_location=device, weights_only=True))
        full_model.eval()
        full_scaler = joblib.load('dist/models/full_model_scaler.joblib')
        full_target_scaler = joblib.load('dist/models/full_model_target_scaler.joblib')
        
        # Baseline-PhysFeat
        phys_model = DigitalTwinModel(input_dim=phys_input_dim, hidden_dim=32, dropout_rate=0.1).to(device)
        phys_model.load_state_dict(torch.load('dist/models/baseline-physfeat.pth', map_location=device, weights_only=True))
        phys_model.eval()
        phys_scaler = joblib.load('dist/models/baseline-physfeat_scaler.joblib')
        phys_target_scaler = joblib.load('dist/models/baseline-physfeat_target_scaler.joblib')
        
        # Baseline-Raw
        raw_model = DigitalTwinModel(input_dim=raw_input_dim, hidden_dim=32, dropout_rate=0.1).to(device)
        raw_model.load_state_dict(torch.load('dist/models/baseline-raw.pth', map_location=device, weights_only=True))
        raw_model.eval()
        raw_scaler = joblib.load('dist/models/baseline-raw_scaler.joblib')
        raw_target_scaler = joblib.load('dist/models/baseline-raw_target_scaler.joblib')
        
        print("Successfully loaded all models and scalers.")
    except Exception as e:
        print(f"Error loading models: {e}. Did you run train.py?")
        return

    # 2. Load Real Dataset to stream
    try:
        df_raw = pd.read_csv('Dataset/turbojet_complete_dataset.csv')
        df_raw.rename(columns={
            'Tamb_K': 'Tamb', 'Pamb_Pa': 'Pamb', 'T2_K': 'T2', 'P2_Pa': 'P2',
            'T3_K': 'T3', 'P3_Pa': 'P3', 'T4_K': 'T4', 'P4_Pa': 'P4',
            'RPM_rev_min': 'RPM', 'FuelFlow_kg_s': 'Fuel_Flow'
        }, inplace=True)
    except FileNotFoundError:
        print("Dataset not found. Cannot stream.")
        return

    print("Initiating Live Stream...")
    
    # Loop through dataset simulating live feed
    for idx, row in df_raw.iterrows():
        row_df = pd.DataFrame([row])
        fuel_flow_g = row['Fuel_Flow'] * 1000.0
        
        # --- PREPARE INPUTS ---
        phys_features = thermo.extract_physics_features(row_df)
        raw_sensor_cols = ['Tamb', 'Pamb', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'RPM', 'Fuel_Flow', 'Altitude_m', 'Mach']
        raw_features = row_df[raw_sensor_cols]
        
        full_x = torch.tensor(full_scaler.transform(phys_features.values), dtype=torch.float32).to(device)
        phys_x = torch.tensor(phys_scaler.transform(phys_features.values), dtype=torch.float32).to(device)
        raw_x = torch.tensor(raw_scaler.transform(raw_features.values), dtype=torch.float32).to(device)
        
        # --- INFERENCE ---
        with torch.no_grad():
            # Full Model (with uncertainty)
            full_stats = full_model.predict_with_uncertainty(full_x, num_samples=10)
            
            # Baseline-PhysFeat
            phys_preds = phys_model(phys_x)
            
            # Baseline-Raw
            raw_preds = raw_model(raw_x)
            
        # --- DENORMALIZE ---
        # Full Model
        comp_h = full_stats[0][0].item() * full_target_scaler.scale_[0] + full_target_scaler.mean_[0]
        comb_h = full_stats[1][0].item() * full_target_scaler.scale_[1] + full_target_scaler.mean_[1]
        turb_h = full_stats[2][0].item() * full_target_scaler.scale_[2] + full_target_scaler.mean_[2]
        overall_h = full_stats[3][0].item() * full_target_scaler.scale_[3] + full_target_scaler.mean_[3]
        overall_std = full_stats[3][1].item() * full_target_scaler.scale_[3]
        thrust = full_stats[4][0].item() * full_target_scaler.scale_[4] + full_target_scaler.mean_[4]
        
        # Deterministic TSFC
        tsfc = fuel_flow_g / (thrust + 1e-6)
        
        # Baseline-PhysFeat
        phys_thrust = phys_preds[4].item() * phys_target_scaler.scale_[4] + phys_target_scaler.mean_[4]
        phys_tsfc = fuel_flow_g / (phys_thrust + 1e-6)
        
        # Baseline-Raw
        raw_thrust = raw_preds[4].item() * raw_target_scaler.scale_[4] + raw_target_scaler.mean_[4]
        raw_tsfc = fuel_flow_g / (raw_thrust + 1e-6)
        
        # --- METRICS ---
        efficiency = phys_features['Comp_Isentropic_Efficiency'].values[0]
        physics_consistency = min(efficiency * 100, 100)
        physics_score = f"{physics_consistency:.1f}%"
        
        # TSFC violation is 0.0% by construction now. We can still send 0.0
        pinn_violation = 0.0
        phys_violation = 0.0
        raw_violation = 0.0
        
        payload = {
            "cycle": row['Cycle'],
            "comp_health": comp_h * 100,
            "comb_health": comb_h * 100,
            "turb_health": turb_h * 100,
            "overall_health": overall_h * 100,
            "thrust": thrust,
            "tsfc": tsfc, # Sending deterministic TSFC
            "uncertainty_overall": overall_std * 100,
            "physics_score": physics_score,
            "pinn_violation": pinn_violation,
            "icarus_violation": phys_violation, # Reusing field for PhysFeat
            "titan_violation": raw_violation,   # Reusing field for Raw
            "op_altitude": row['Altitude_m'],
            "op_mach": row['Mach'],
            "op_tamb": row['Tamb'],
            "op_pamb": row['Pamb'],
            "op_rpm": row['RPM'],
            "op_fuel": row['Fuel_Flow']
        }
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print(f"Cycle {row['Cycle']} | Health: {overall_h*100:.1f}% | TSFC: Structurally enforced (Δthrust-driven)")
        except requests.exceptions.RequestException as e:
            print(f"Network Warning: Could not push telemetry to Dashboard. ({type(e).__name__})")
            time.sleep(2)
            continue
            
        time.sleep(0.5)

    print("Engine Run Complete.")

if __name__ == "__main__":
    stream_telemetry()
