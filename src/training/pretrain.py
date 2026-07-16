import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import sys

# Add parent dir to path so we can import src.models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.transfer import ContrastivePretrainingModel

def contrastive_loss(h1, h2, temperature=0.1):
    """
    InfoNCE Contrastive Loss.
    h1 and h2 are shape (batch_size, hidden_dim)
    """
    # Normalize representations along hidden dim
    h1 = F.normalize(h1, dim=-1)
    h2 = F.normalize(h2, dim=-1)
    
    # Compute similarity matrix (batch_size, batch_size)
    # sim[i, j] = similarity between h1_i and h2_j
    sim_12 = torch.matmul(h1, h2.t()) / temperature
    sim_21 = torch.matmul(h2, h1.t()) / temperature
    
    batch_size = h1.size(0)
    labels = torch.arange(batch_size, device=h1.device)
    
    # Cross entropy loss on similarity matrix
    loss_12 = F.cross_entropy(sim_12, labels)
    loss_21 = F.cross_entropy(sim_21, labels)
    
    return (loss_12 + loss_21) / 2

def main():
    data_path = 'Dataset/processed/ncmapss_pairs_128.npz'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run ncmapss_extractor.py first.")
        return
        
    print("Loading data...")
    data = np.load(data_path)
    X1_np = data['X1']
    X2_np = data['X2']
    
    # Global Z-Score Normalization for Pretraining
    # We compute stats over the entire X1 dataset to give the adapter a consistent scale
    mean = X1_np.mean(axis=(0, 1), keepdims=True)
    std = X1_np.std(axis=(0, 1), keepdims=True) + 1e-6
    
    X1_np = (X1_np - mean) / std
    X2_np = (X2_np - mean) / std
    
    print(f"Data shape: {X1_np.shape}. Total pairs: {len(X1_np)}")
    
    X1_t = torch.tensor(X1_np, dtype=torch.float32)
    X2_t = torch.tensor(X2_np, dtype=torch.float32)
    
    dataset = TensorDataset(X1_t, X2_t)
    # Shuffle is crucial so negatives in the batch are from different flights
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ContrastivePretrainingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 30
    best_loss = float('inf')
    
    os.makedirs('models', exist_ok=True)
    save_path = 'models/ncmapss_pretrained_encoder.pth'
    
    print("Starting Pretraining...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for b_x1, b_x2 in pbar:
            b_x1, b_x2 = b_x1.to(device), b_x2.to(device)
            
            optimizer.zero_grad()
            
            h1 = model(b_x1)
            h2 = model(b_x2)
            
            loss = contrastive_loss(h1, h2)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save only the shared encoder (the adapter is N-CMAPSS specific)
            torch.save(model.encoder.state_dict(), save_path)
            print(f"--> Saved best encoder weights to {save_path}")
            
    print("Pretraining complete.")

if __name__ == '__main__':
    main()
