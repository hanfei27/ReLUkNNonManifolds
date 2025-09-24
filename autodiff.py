
import torch
from equations.sphere_ops import tangential_grad, lb_operator, proj_matrix


def value_grad_hess(f, x, need_hess=False):
    x = x.requires_grad_(True)
    y = f(x)  # (...,1)
    g_tan = tangential_grad(f, x)  # intrinsic grad
    if not need_hess:
        return y, g_tan, None
    # Hessian on the surface is subtle; we approximate by projected ambient Hessian action on tangent directions:
    # H_T ≈ P (∇^2 f) P  (works well for tests up to s=2 at moderate sizes)
    grads = torch.autograd.grad(y.squeeze(-1), x, torch.ones_like(y.squeeze(-1)), create_graph=True)[0]
    d = grads.shape[-1]
    cols = []
    for i in range(d):
        gi = grads[..., i]
        gi_x = torch.autograd.grad(gi, x, torch.ones_like(gi), create_graph=True)[0]
        cols.append(gi_x.unsqueeze(-1))
    H = torch.cat(cols, dim=-1)  # (...,d,d)
    P = proj_matrix(x)
    Ht = torch.einsum("...ij,...jk,...kl->...il", P, H, P)  # projected Hessian
    return y, g_tan, Ht


def laplace_beltrami(f, x):
    return lb_operator(f, x).unsqueeze(-1)
