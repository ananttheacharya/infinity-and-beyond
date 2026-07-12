import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNDigitalTwin(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for Digital Twin.
    Incorporates MC Dropout for Uncertainty Quantification.
    Hybrid output: 
        1. RUL (Remaining Useful Life) Regressor
        2. Health Status (Healthy, Degrading, Critical) Classifier
    """
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
        super(PINNDigitalTwin, self).__init__()
        
        # Shared Feature Extractor (Backbone)
        self.shared_fc1 = nn.Linear(input_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.shared_fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # MC Dropout Layer - we keep this active even during inference for uncertainty estimation
        self.mc_dropout = nn.Dropout(p=dropout_rate)
        
        # Head 1: RUL Regressor
        self.rul_head_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.rul_head_out = nn.Linear(hidden_dim // 2, 1)
        
        # Head 2: Health Risk Classifier (3 classes)
        self.risk_head_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.risk_head_out = nn.Linear(hidden_dim // 2, 3)

    def forward(self, x):
        # Shared layers
        x = F.relu(self.shared_fc1(x))
        x = self.mc_dropout(x)
        x = F.relu(self.shared_fc2(x))
        x = self.mc_dropout(x)
        x = F.relu(self.shared_fc3(x))
        x = self.mc_dropout(x)
        
        # RUL Head
        rul_features = F.relu(self.rul_head_fc1(x))
        rul_features = self.mc_dropout(rul_features)
        rul_pred = F.softplus(self.rul_head_out(rul_features)) # Softplus ensures positive RUL
        
        # Risk Head
        risk_features = F.relu(self.risk_head_fc1(x))
        risk_features = self.mc_dropout(risk_features)
        risk_pred = self.risk_head_out(risk_features) # CrossEntropyLoss will handle softmax
        
        return rul_pred, risk_pred
        
    def predict_with_uncertainty(self, x, num_samples=50):
        """
        Runs multiple forward passes with Dropout enabled to calculate 
        predictive mean and variance (MC Dropout).
        """
        # Ensure dropout is active
        self.train() 
        
        rul_preds = []
        for _ in range(num_samples):
            rul_pred, _ = self.forward(x)
            rul_preds.append(rul_pred)
            
        rul_preds = torch.stack(rul_preds)
        
        # Calculate mean and standard deviation (uncertainty)
        rul_mean = rul_preds.mean(dim=0)
        rul_std = rul_preds.std(dim=0)
        
        # Reset to eval mode ideally, but leaving as is for simplicity
        return rul_mean, rul_std
