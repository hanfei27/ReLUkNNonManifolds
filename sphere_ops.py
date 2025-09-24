
import torch


def proj_matrix(x):
    """Projection onto the tangent space at x for a unit sphere in R^d."""
    d = x.shape[-1]
    I = torch.eye(d, device=x.device, dtype=x.dtype).expand(x.shape[:-1] + (d, d))
    xxT = x.unsqueeze(-1) * x.unsqueeze(-2)
    return I - xxT


def grad(f, x):
    """Ambient gradient via autograd; returns (..., d)."""
    x = x.requires_grad_(True)
    y = f(x)
    if y.ndim > 1:
        y = y.squeeze(-1)
    g = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return g


def laplacian(f, x):
    """Ambient Euclidean Laplacian Δ f."""
    x = x.requires_grad_(True)
    y = f(x).squeeze(-1)
    grads = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    lap = 0.0
    d = grads.shape[-1]
    for i in range(d):
        gi = grads[..., i]
        gi_x = torch.autograd.grad(gi, x, grad_outputs=torch.ones_like(gi), create_graph=True)[0][..., i]
        lap = lap + gi_x
    return lap


def lb_operator(f, x):
    """Laplace–Beltrami via embedding formula on S^{d-1} ⊂ R^d.

    Uses Δ_{S^{d-1}} f = Δ f - (x·∇)^2 f - (d-2)(x·∇ f), evaluated at |x|=1.
    """
    x = x.requires_grad_(True)
    g = grad(f, x)
    x_dot_grad = (x * g).sum(dim=-1)

    def phi(z):
        return (z * grad(f, z)).sum(dim=-1, keepdim=True)

    x_dot_grad_sq = (grad(phi, x) * x).sum(dim=-1)
    lap = laplacian(f, x)
    ambient_dim = x.shape[-1]
    return lap - x_dot_grad_sq - (ambient_dim - 2) * x_dot_grad


def tangential_grad(f, x):
    """Intrinsic gradient obtained by projecting the ambient gradient."""
    g = grad(f, x)
    P = proj_matrix(x)
    return torch.einsum("...ij,...j->...i", P, g)
