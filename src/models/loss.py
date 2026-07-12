import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """
    Custom Loss Function for the Physics-Informed Neural Network.
    Total Loss = α * MSE(RUL) + β * CrossEntropy(Health) + γ * Physics_Violation_Penalty
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=2.0):
        super(PhysicsInformedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, rul_pred, rul_target, risk_pred, risk_target, physics_features):
        """
        rul_pred, rul_target: Tensors for Remaining Useful Life
        risk_pred, risk_target: Tensors for Health Status (Logits, Class Index)
        physics_features: Tensor of calculated thermodynamic features 
                          e.g., [PR_comp, Isentropic_Efficiency, ...]
        """
        # 1. RUL Loss (Regression)
        loss_rul = self.mse_loss(rul_pred.squeeze(), rul_target)
        
        # 2. Risk Classification Loss
        loss_risk = self.ce_loss(risk_pred, risk_target)
        
        # 3. Physics Violation Penalty
        # For example, Isentropic Efficiency (index 2) should never be > 1.0 (100%) or < 0
        # If it is, we add a massive penalty. 
        # In practice, this would also use predictions of physical states, but here we enforce 
        # that the network doesn't predict states mapping to impossible physics.
        # Assuming physics_features[:, 2] is efficiency:
        
        # If efficiency > 1.0, penalty = efficiency - 1.0
        eff = physics_features[:, 2]
        penalty_high = torch.relu(eff - 1.0)
        # If efficiency < 0.0, penalty = 0.0 - efficiency
        penalty_low = torch.relu(-eff)
        
        physics_penalty = torch.mean(penalty_high + penalty_low)
        
        # Total Loss
        total_loss = (self.alpha * loss_rul) + (self.beta * loss_risk) + (self.gamma * physics_penalty)
        
        return total_loss, loss_rul, loss_risk, physics_penalty
