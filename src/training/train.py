import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Adjust imports for running from root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.dataset import load_and_merge_data, get_engine_split, prepare_dataloaders
from src.models.pinn import DigitalTwinModel
from src.models.loss import PhysicsConstrainedLoss
import joblib

def train_model(model_name, use_physics, alpha, beta_health, df_train, df_val, train_idx, val_idx, device):
    print(f"\n=== Training Variant: {model_name} ===")
    
    train_loader, val_loader, scaler, target_scaler, feature_cols = prepare_dataloaders(
        pd.concat([df_train, df_val], ignore_index=True), 
        list(range(len(df_train))), 
        list(range(len(df_train), len(df_train) + len(df_val))), 
        batch_size=64, 
        use_physics_features=use_physics
    )
    
    input_dim = len(feature_cols)
    model = DigitalTwinModel(input_dim=input_dim, hidden_dim=32, dropout_rate=0.1).to(device)
    criterion = PhysicsConstrainedLoss(alpha=alpha, beta_health=beta_health)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    target_mean = torch.tensor(target_scaler.mean_, dtype=torch.float32).to(device)
    target_scale = torch.tensor(target_scaler.scale_, dtype=torch.float32).to(device)

    epochs = 300
    patience = 20
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch_x = batch[0].to(device)
            batch_targets = batch[1].to(device)
            fuel_flow_g = batch[2].to(device) * 1000.0 # Convert kg/s to g/s
            
            optimizer.zero_grad()
            preds = model(batch_x)
            
            total_loss, mse_total, tsfc_cons, health_cons = criterion(preds, batch_targets, fuel_flow_g, target_mean, target_scale)
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch[0].to(device)
                batch_targets = batch[1].to(device)
                fuel_flow_g = batch[2].to(device) * 1000.0
                
                preds = model(batch_x)
                total_loss, mse_total, _, _ = criterion(preds, batch_targets, fuel_flow_g, target_mean, target_scale)
                val_loss += mse_total.item()
                
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model and scaler
    os.makedirs('dist/models', exist_ok=True)
    model_name_lower = model_name.replace(" ", "_").lower()
    torch.save(model.state_dict(), f'dist/models/{model_name_lower}.pth')
    joblib.dump(scaler, f'dist/models/{model_name_lower}_scaler.joblib')
    joblib.dump(target_scaler, f'dist/models/{model_name_lower}_target_scaler.joblib')
    print(f"Saved best {model_name} model and scalers.")
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df_merged = load_and_merge_data()
    # Test engines: 9, 10
    df_train_val, df_test = get_engine_split(df_merged, test_engines=[9, 10])
    
    # Validation engines: 7, 8
    df_train, df_val = get_engine_split(df_train_val, test_engines=[7, 8])
    
    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")
    
    variants = [
        {"name": "Baseline-Raw", "use_physics": False, "alpha": 1.0, "beta_health": 0.0},
        {"name": "Baseline-PhysFeat", "use_physics": True, "alpha": 1.0, "beta_health": 0.0},
        {"name": "Full Model", "use_physics": True, "alpha": 1.0, "beta_health": 1.0}
    ]
    
    for v in variants:
        train_model(
            v["name"], v["use_physics"], v["alpha"], v["beta_health"], 
            df_train, df_val, list(range(len(df_train))), list(range(len(df_train), len(df_train) + len(df_val))),
            device
        )

if __name__ == "__main__":
    main()
