
import torch

def weighted_l2(y, w):
    return torch.sqrt((w*(y.squeeze(-1)**2)).sum())

def weighted_mse(y, w):
    # weighted mean squared error: sum(w * y^2) / sum(w)
    num = (w*(y.squeeze(-1)**2)).sum()
    den = w.sum() + 1e-12
    return (num/den)

def weighted_rmse(y, w):
    return torch.sqrt(weighted_mse(y, w))

def rel_l2(err, ref, w):
    num = weighted_l2(err, w)
    den = weighted_l2(ref, w) + 1e-12
    return (num/den).item()

def max_abs(y):
    return y.abs().max().item()
