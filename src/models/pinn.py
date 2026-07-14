import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitalTwinModel(nn.Module):
    """
    Digital Twin Model Architecture.
    Used for Baseline-Raw, Baseline-PhysFeat, and Full Model (PCMN).
    Multi-head output for Competition Deliverables: 
        1. Compressor Health (0-1)
        2. Combustor Health (0-1)
        3. Turbine Health (0-1)
        4. Overall Health (0-1)
        5. Engine Thrust (N)
    (TSFC is computed deterministically outside the network)
    """
    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0.1):
        super(DigitalTwinModel, self).__init__()
        
        # Shared Feature Extractor (Backbone)
        self.shared_fc1 = nn.Linear(input_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.shared_fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # MC Dropout Layer - we keep this active even during inference for uncertainty estimation
        self.mc_dropout = nn.Dropout(p=dropout_rate)
        
        # Output Heads (All share the backbone)
        self.compressor_head = nn.Linear(hidden_dim, 1)
        self.combustor_head = nn.Linear(hidden_dim, 1)
        self.turbine_head = nn.Linear(hidden_dim, 1)
        self.overall_health_head = nn.Linear(hidden_dim, 1)
        self.thrust_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Shared layers
        x = F.relu(self.shared_fc1(x))
        x = self.mc_dropout(x)
        x = F.relu(self.shared_fc2(x))
        x = self.mc_dropout(x)
        x = F.relu(self.shared_fc3(x))
        x = self.mc_dropout(x)
        
        # Subsystem & Overall Health (Z-score normalized)
        comp_h = self.compressor_head(x)
        comb_h = self.combustor_head(x)
        turb_h = self.turbine_head(x)
        overall_h = self.overall_health_head(x)
        
        # Performance Metrics (Z-score normalized)
        thrust = self.thrust_head(x)
        
        return comp_h, comb_h, turb_h, overall_h, thrust
        
    def predict_with_uncertainty(self, x, num_samples=30):
        """
        Runs multiple forward passes with Dropout enabled to calculate 
        predictive mean and variance (MC Dropout).
        Returns means and stds for all 5 outputs.
        """
        self.train() # Ensure dropout is active
        
        preds_comp, preds_comb, preds_turb = [], [], []
        preds_overall, preds_thrust = [], []
        
        with torch.no_grad():
            for _ in range(num_samples):
                comp, comb, turb, overall, thrust = self.forward(x)
                preds_comp.append(comp)
                preds_comb.append(comb)
                preds_turb.append(turb)
                preds_overall.append(overall)
                preds_thrust.append(thrust)
            
        # Helper to calculate mean and std
        def get_stats(tensor_list):
            stacked = torch.stack(tensor_list)
            return stacked.mean(dim=0), stacked.std(dim=0)
            
        return (
            get_stats(preds_comp), get_stats(preds_comb), get_stats(preds_turb),
            get_stats(preds_overall), get_stats(preds_thrust)
        )
