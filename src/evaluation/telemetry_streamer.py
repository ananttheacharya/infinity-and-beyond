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
    phys_features = thermo.extract_physics_features(df_dummy).drop(columns=['Altitude_m', 'Mach'], errors='ignore')
    raw_input_dim = 12 # sensor cols
    combined_input_dim = raw_input_dim + phys_features.shape[1]
    
    seq_len = 5
    
    # 1. Load the Models
    try:
        # Full Model (GRU N=5)
        full_model = DigitalTwinModel(input_dim=combined_input_dim, hidden_dim=32, dropout_rate=0.1, model_type='gru').to(device)
        full_model.load_state_dict(torch.load('dist/models/full_model.pth', map_location=device, weights_only=True))
        full_model.eval()
        full_scaler = joblib.load('dist/models/full_model_scaler.joblib')
        full_target_scaler = joblib.load('dist/models/full_model_target_scaler.joblib')
        
        # PhysFeat-Combined (MLP N=1)
        phys_model = DigitalTwinModel(input_dim=combined_input_dim, hidden_dim=32, dropout_rate=0.1, model_type='mlp').to(device)
        phys_model.load_state_dict(torch.load('dist/models/physfeat-combined.pth', map_location=device, weights_only=True))
        phys_model.eval()
        phys_scaler = joblib.load('dist/models/physfeat-combined_scaler.joblib')
        phys_target_scaler = joblib.load('dist/models/physfeat-combined_target_scaler.joblib')
        
        # Baseline-Raw (MLP N=1)
        raw_model = DigitalTwinModel(input_dim=raw_input_dim, hidden_dim=32, dropout_rate=0.1, model_type='mlp').to(device)
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
    
    # State buffers
    full_buffer = []
    current_engine = None
    
    # Loop through dataset simulating live feed
    for idx, row in df_raw.iterrows():
        row_df = pd.DataFrame([row])
        fuel_flow_g = row['Fuel_Flow'] * 1000.0
        
        # If new engine, clear buffer
        if len(full_buffer) > 0 and current_engine != row['EngineID']:
            full_buffer = []
        current_engine = row['EngineID']
        
        # --- PREPARE INPUTS ---
        phys_features = thermo.extract_physics_features(row_df).drop(columns=['Altitude_m', 'Mach'], errors='ignore')
        raw_sensor_cols = ['Tamb', 'Pamb', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'RPM', 'Fuel_Flow', 'Altitude_m', 'Mach']
        raw_features = row_df[raw_sensor_cols]
        
        combined_features = pd.concat([raw_features.reset_index(drop=True), phys_features.reset_index(drop=True)], axis=1)
        
        full_x_step = full_scaler.transform(combined_features.values)
        phys_x_step = phys_scaler.transform(combined_features.values)
        raw_x_step = raw_scaler.transform(raw_features.values)
        
        full_buffer.append(full_x_step[0])
        
        # If we don't have enough history, output cold-start
        if len(full_buffer) < seq_len:
            payload = {
                "cycle": row['Cycle'],
                "status": "Insufficient History",
                "message": f"Buffering sequences: {len(full_buffer)}/{seq_len}",
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
                    print(f"Cycle {row['Cycle']} | Buffering {len(full_buffer)}/{seq_len}...")
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.1)
            continue
            
        if len(full_buffer) > seq_len:
            full_buffer.pop(0)
            
        full_x = torch.tensor(np.array([full_buffer]), dtype=torch.float32).to(device)
        phys_x = torch.tensor(phys_x_step, dtype=torch.float32).to(device)
        raw_x = torch.tensor(raw_x_step, dtype=torch.float32).to(device)
        
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
        efficiency = phys_features['Comp_Isentropic_Efficiency'].values[0] if 'Comp_Isentropic_Efficiency' in phys_features else 0.8
        physics_consistency = min(efficiency * 100, 100)
        physics_score = f"{physics_consistency:.1f}%"
        
        # TSFC violation is 0.0% by construction now
        pinn_violation = 0.0
        phys_violation = 0.0
        raw_violation = 0.0
        
        payload = {
            "cycle": row['Cycle'],
            "status": "Active",
            "comp_health": comp_h * 100,
            "comb_health": comb_h * 100,
            "turb_health": turb_h * 100,
            "overall_health": overall_h * 100,
            "thrust": thrust,
            "tsfc": tsfc,
            "uncertainty_overall": overall_std * 100,
            "physics_score": physics_score,
            "pinn_violation": pinn_violation,
            "icarus_violation": phys_violation, 
            "titan_violation": raw_violation,   
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
