import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """
    Custom Loss Function for the Physics-Informed Neural Network.
    Calculates MSE for 6 regression targets + Physics Penalty.
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super(PhysicsInformedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, targets, physics_features):
        """
        preds: Tuple of (comp, comb, turb, overall, thrust, tsfc)
        targets: Tuple of (comp, comb, turb, overall, thrust, tsfc)
        physics_features: Tensor of calculated thermodynamic features 
        """
        comp_p, comb_p, turb_p, overall_p, thrust_p, tsfc_p = preds
        comp_t, comb_t, turb_t, overall_t, thrust_t, tsfc_t = targets
        
        # Calculate MSE for all 6 targets
        loss_comp = self.mse_loss(comp_p.squeeze(), comp_t)
        loss_comb = self.mse_loss(comb_p.squeeze(), comb_t)
        loss_turb = self.mse_loss(turb_p.squeeze(), turb_t)
        loss_overall = self.mse_loss(overall_p.squeeze(), overall_t)
        loss_thrust = self.mse_loss(thrust_p.squeeze(), thrust_t)
        loss_tsfc = self.mse_loss(tsfc_p.squeeze(), tsfc_t)
        
        mse_total = loss_comp + loss_comb + loss_turb + loss_overall + loss_thrust + loss_tsfc
        
        # Physics Violation Penalty (e.g., Isentropic Efficiency bounds)
        eff = physics_features[:, 2] # Assuming index 2 is efficiency
        penalty_high = torch.relu(eff - 1.0)
        penalty_low = torch.relu(-eff)
        physics_penalty = torch.mean(penalty_high + penalty_low)
        
        # Total Loss
        total_loss = (self.alpha * mse_total) + (self.gamma * physics_penalty)
        
        return total_loss, mse_total, physics_penalty
