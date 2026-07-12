import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# Adjust imports for running from root directory
import sys
sys.path.append('.') 
from src.data_pipeline.thermodynamics import ThermodynamicsEngine
from src.models.pinn import PINNDigitalTwin
from src.models.loss import PhysicsInformedLoss

def load_real_data(csv_path='Dataset/turbojet_complete_dataset.csv'):
    """
    Loads the real complete dataset containing sensors and targets.
    """
    df = pd.read_csv(csv_path)
    
    # Rename columns to match ThermodynamicsEngine expectations
    df.rename(columns={
        'Tamb_K': 'Tamb', 'Pamb_Pa': 'Pamb', 'T2_K': 'T2', 'P2_Pa': 'P2',
        'T3_K': 'T3', 'P3_Pa': 'P3', 'T4_K': 'T4', 'P4_Pa': 'P4',
        'RPM_rev_min': 'RPM', 'FuelFlow_kg_s': 'Fuel_Flow'
    }, inplace=True)
    
    # Extract target columns
    comp_h = df['CompressorHealth'].values
    comb_h = df['CombustorHealth'].values
    turb_h = df['TurbineHealth'].values
    overall_h = df['OverallHealth'].values
    thrust = df['Thrust_N'].values
    tsfc = df['TSFC_g_N_s'].values
    
    targets = (comp_h, comb_h, turb_h, overall_h, thrust, tsfc)
    return df, targets

def train():
    print("=== Training 6-Headed PINN on Real Data ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading Dataset/turbojet_complete_dataset.csv...")
    try:
        df_raw, targets_np = load_real_data()
    except FileNotFoundError:
        print("Error: Dataset/turbojet_complete_dataset.csv not found.")
        return

    thermo_engine = ThermodynamicsEngine()
    df_phys = thermo_engine.extract_physics_features(df_raw)
    
    # Convert to Tensors
    X = torch.tensor(df_phys.values, dtype=torch.float32)
    t_comp = torch.tensor(targets_np[0], dtype=torch.float32)
    t_comb = torch.tensor(targets_np[1], dtype=torch.float32)
    t_turb = torch.tensor(targets_np[2], dtype=torch.float32)
    t_over = torch.tensor(targets_np[3], dtype=torch.float32)
    t_thrust = torch.tensor(targets_np[4], dtype=torch.float32)
    t_tsfc = torch.tensor(targets_np[5], dtype=torch.float32)

    dataset = TensorDataset(X, t_comp, t_comb, t_turb, t_over, t_thrust, t_tsfc)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Initialize Model
    input_dim = X.shape[1]
    model = PINNDigitalTwin(input_dim=input_dim, hidden_dim=128, dropout_rate=0.3).to(device)
    criterion = PhysicsInformedLoss(alpha=1.0, gamma=5.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    epochs = 10
    print(f"Starting training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        phys_violations = 0
        
        for batch in dataloader:
            batch_x = batch[0].to(device)
            batch_targets = tuple(t.to(device) for t in batch[1:])
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(batch_x)
            
            # Calculate loss
            total_loss, mse_total, phys_penalty = criterion(preds, batch_targets, batch_x)
            
            # Backprop
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            if phys_penalty.item() > 0:
                phys_violations += 1
                
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f} - Physics Violations in Batches: {phys_violations}")

    # Save Model
    os.makedirs('dist/models', exist_ok=True)
    torch.save(model.state_dict(), 'dist/models/pinn_model.pth')
    print("Training complete! Model saved to dist/models/pinn_model.pth")

if __name__ == "__main__":
    train()
