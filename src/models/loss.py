import torch
import torch.nn as nn

class PhysicsConstrainedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta_health=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta_health = beta_health
        self.mse = nn.MSELoss()

    def forward(self, preds, targets, fuel_flow_g, target_mean=None, target_scale=None):
        comp_p, comb_p, turb_p, overall_p, thrust_p = preds
        comp_t = targets[:, 0]
        comb_t = targets[:, 1]
        turb_t = targets[:, 2]
        overall_t = targets[:, 3]
        thrust_t = targets[:, 4]

        mse_total = (self.mse(comp_p.squeeze(), comp_t) + 
                     self.mse(comb_p.squeeze(), comb_t) +
                     self.mse(turb_p.squeeze(), turb_t) +
                     self.mse(overall_p.squeeze(), overall_t) +
                     self.mse(thrust_p.squeeze(), thrust_t))

        if target_mean is not None and target_scale is not None:
            # Denormalize predictions to enforce physics constraints
            comp_real = comp_p.squeeze() * target_scale[0] + target_mean[0]
            comb_real = comb_p.squeeze() * target_scale[1] + target_mean[1]
            turb_real = turb_p.squeeze() * target_scale[2] + target_mean[2]
            thrust_real = thrust_p.squeeze() * target_scale[4] + target_mean[4]
            # CONSTRAINT: Overall health must be explained by subsystem health
            expected_overall_real = (0.40 * comp_real + 0.30 * turb_real + 0.30 * comb_real)
            expected_overall_norm = (expected_overall_real - target_mean[3]) / target_scale[3]
            health_consistency = self.mse(overall_p.squeeze(), expected_overall_norm)
        else:
            # Fallback
            expected_overall = (0.40 * comp_p.squeeze() + 0.30 * turb_p.squeeze() + 0.30 * comb_p.squeeze())
            health_consistency = self.mse(overall_p.squeeze(), expected_overall)

        total = (self.alpha * mse_total + 
                 self.beta_health * health_consistency)
                 
        return total, mse_total, 0.0, health_consistency
