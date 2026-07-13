import pandas as pd
import numpy as np
import torch
import torch.nn as pd_nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Define inputs and targets
INPUT_COLS = ['Altitude_m', 'Mach', 'Tamb_K', 'Pamb_Pa', 'RPM_rev_min', 'FuelFlow_kg_s', 
              'P2_Pa', 'T2_K', 'P3_Pa', 'T3_K', 'P4_Pa', 'T4_K']
TARGET_COLS = ['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 
               'OverallHealth', 'Thrust_N', 'TSFC_g_N_s']

def train_xgboost():
    print("Training XGBoost Baseline (Project Titan)...")
    from xgboost import XGBRegressor
    
    df = pd.read_csv('Dataset/turbojet_complete_dataset.csv')
    X = df[INPUT_COLS].values
    y = df[TARGET_COLS].values
    
    # Standardize inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train multi-output XGBoost
    from sklearn.multioutput import MultiOutputRegressor
    xgb = XGBRegressor(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42)
    model = MultiOutputRegressor(xgb)
    model.fit(X_scaled, y)
    
    # Save the model and scaler
    joblib.dump(model, 'src/models/xgboost_titan.joblib')
    joblib.dump(scaler, 'src/models/xgb_scaler.joblib')
    print("XGBoost training complete and saved.")

class GRUBaseline(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUBaseline, self).__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq, features)
        out, _ = self.gru(x)
        # take the last output
        out = self.fc(out[:, -1, :])
        return out

def train_gru():
    print("Training GRU Baseline (Project Icarus)...")
    df = pd.read_csv('Dataset/turbojet_complete_dataset.csv')
    X = df[INPUT_COLS].values
    y = df[TARGET_COLS].values
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Reshape for GRU: sequence length 1
    X_seq = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = GRUBaseline(len(INPUT_COLS), 64, len(TARGET_COLS))
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(10): # Quick train for showdown
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    torch.save(model.state_dict(), 'src/models/gru_icarus.pt')
    joblib.dump(scaler_X, 'src/models/gru_scaler.joblib')
    print("GRU training complete and saved.")

if __name__ == '__main__':
    train_xgboost()
    train_gru()
