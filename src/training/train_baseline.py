import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Adjust imports
import sys
sys.path.append('.') 

def generate_mock_raw_data(num_samples=1000):
    """
    Baseline gets RAW data, not thermodynamic features.
    """
    np.random.seed(42)
    # 10 raw sensor features
    X = np.random.uniform(0, 1, (num_samples, 10))
    rul = np.random.uniform(0, 100, num_samples)
    return X, rul

class BaselineLSTM(nn.Module):
    """
    Replication of the standard data-driven "Black Box" model from competitor repos.
    They typically use standard LSTMs or MLPs on raw scaled data.
    """
    def __init__(self, input_dim):
        super(BaselineLSTM, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

def train():
    print("=== Training Competitor Baseline Model (Raw Data MLP/LSTM) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load RAW Data (No Physics Engine)
    print("Loading RAW sensor data...")
    X_raw, rul_target = generate_mock_raw_data(2000)
    
    X = torch.tensor(X_raw, dtype=torch.float32)
    y_rul = torch.tensor(rul_target, dtype=torch.float32)

    dataset = TensorDataset(X, y_rul)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Initialize Baseline Model
    model = BaselineLSTM(input_dim=10).to(device)
    criterion = nn.MSELoss() # Standard MSE, no physics penalty
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    epochs = 10
    print(f"Starting training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_x, batch_rul in dataloader:
            batch_x, batch_rul = batch_x.to(device), batch_rul.to(device)
            
            optimizer.zero_grad()
            rul_pred = model(batch_x)
            
            loss = criterion(rul_pred.squeeze(), batch_rul)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{epochs} - Standard MSE Loss: {epoch_loss/len(dataloader):.4f}")

    # Save Model
    os.makedirs('dist/models', exist_ok=True)
    torch.save(model.state_dict(), 'dist/models/baseline_model.pth')
    print("Training complete! Baseline model saved to dist/models/baseline_model.pth")

if __name__ == "__main__":
    train()
