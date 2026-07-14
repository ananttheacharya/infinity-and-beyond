import torch
import numpy as np

def compute_tsfc_violation(predicted_tsfc, theoretical_tsfc):
    """
    Computes the TSFC violation percentage.
    Works for both scalar values and PyTorch/NumPy arrays.
    """
    if isinstance(theoretical_tsfc, torch.Tensor):
        zero_mask = theoretical_tsfc <= 0
        theoretical_tsfc_safe = theoretical_tsfc.clone()
        theoretical_tsfc_safe[zero_mask] = 1e-6
        violation = torch.abs(predicted_tsfc - theoretical_tsfc) / theoretical_tsfc_safe * 100.0
        violation[zero_mask] = 0.0
        return violation.mean().item()
    elif isinstance(theoretical_tsfc, np.ndarray):
        zero_mask = theoretical_tsfc <= 0
        theoretical_tsfc_safe = np.copy(theoretical_tsfc)
        theoretical_tsfc_safe[zero_mask] = 1e-6
        violation = np.abs(predicted_tsfc - theoretical_tsfc) / theoretical_tsfc_safe * 100.0
        violation[zero_mask] = 0.0
        return np.mean(violation)
    else:
        if theoretical_tsfc <= 0:
            return 0.0
        return abs(predicted_tsfc - theoretical_tsfc) / theoretical_tsfc * 100.0
