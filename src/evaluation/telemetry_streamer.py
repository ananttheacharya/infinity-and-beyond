import time
import requests
import sys
import os
import torch
import pandas as pd

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.thermodynamics import ThermodynamicsEngine
from src.models.pinn import PINNDigitalTwin

def stream_telemetry():
    print("Starting LIVE Physics-Informed Digital Twin Telemetry Streamer...")
    url = "http://localhost:3000/api/telemetry"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")
    
    # 1. Load the Model
    try:
        # Determine input dim by running a dummy sample through ThermodynamicsEngine
        thermo = ThermodynamicsEngine()
        df_dummy = pd.DataFrame({'Tamb': [288], 'Pamb': [101325], 'T2': [350], 'P2': [150000], 'T3': [900], 'P3': [900000], 'T4': [700], 'P4': [400000], 'RPM': [9000], 'Fuel_Flow': [1.0]})
        input_dim = thermo.extract_physics_features(df_dummy).shape[1]
        
        model = PINNDigitalTwin(input_dim=input_dim, hidden_dim=128).to(device)
        model.load_state_dict(torch.load('dist/models/pinn_model.pth', map_location=device))
        model.eval()
        print("Successfully loaded PINN model weights.")
    except Exception as e:
        print(f"Error loading model: {e}. Did you run orchestrator.py?")
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
    
    # We will loop through the dataset simulating a live feed
    for idx, row in df_raw.iterrows():
        # Prepare input
        row_df = pd.DataFrame([row])
        features = thermo.extract_physics_features(row_df)
        x_tensor = torch.tensor(features.values, dtype=torch.float32).to(device)
        
        # Inference with Uncertainty (MC Dropout)
        predictions = model.predict_with_uncertainty(x_tensor, num_samples=10)
        
        comp_h, comp_std = predictions[0][0].item(), predictions[0][1].item()
        comb_h, comb_std = predictions[1][0].item(), predictions[1][1].item()
        turb_h, turb_std = predictions[2][0].item(), predictions[2][1].item()
        overall_h, overall_std = predictions[3][0].item(), predictions[3][1].item()
        thrust, thrust_std = predictions[4][0].item(), predictions[4][1].item()
        tsfc, tsfc_std = predictions[5][0].item(), predictions[5][1].item()
        
        # Calculate Physics Consistency (e.g. Isentropic Efficiency)
        efficiency = features['Comp_Isentropic_Efficiency'].values[0]
        physics_score = f"{min(efficiency * 100, 100):.1f}%" # Bound to 100% for display
        
        payload = {
            "cycle": row['Cycle'],
            "comp_health": comp_h * 100,
            "comb_health": comb_h * 100,
            "turb_health": turb_h * 100,
            "overall_health": overall_h * 100,
            "thrust": thrust,
            "tsfc": tsfc,
            "uncertainty_overall": overall_std * 100,
            "physics_score": physics_score
        }
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print(f"Cycle {row['Cycle']} | Overall: {overall_h*100:.1f}% | Thrust: {thrust:.0f}N | Physics: {physics_score}")
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Node.js server. Is it running on port 3000?")
            time.sleep(2)
            continue
            
        time.sleep(0.5) # 2 updates per second

    print("Engine Run Complete.")

if __name__ == "__main__":
    stream_telemetry()
