import math
import torch


def torus_surface_area(R: float, r: float) -> float:
    """Surface area of a standard torus with major radius R and minor radius r."""
    R = float(R)
    r = float(r)
    return 4.0 * math.pi ** 2 * R * r


def torus_embed(u: torch.Tensor, v: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """Embed angle coordinates (u, v) into R^3 for a standard torus."""
    R = float(R)
    r = float(r)
    cos_u = torch.cos(u)
    sin_u = torch.sin(u)
    cos_v = torch.cos(v)
    sin_v = torch.sin(v)
    x = (R + r * cos_v) * cos_u
    y = (R + r * cos_v) * sin_u
    z = r * sin_v
    return torch.stack([x, y, z], dim=-1)


def torus_angles(x: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """Recover angle coordinates (u, v) from embedded points on the torus."""
    R = float(R)
    r = float(r)
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]
    rho = torch.sqrt(x0 * x0 + x1 * x1).clamp_min(1e-12)
    u = torch.atan2(x1, x0)
    v = torch.atan2(x2, rho - R)
    return torch.stack([u, v], dim=-1)


def torus_area_element(v: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """Magnitude of the cross product X_u × X_v for torus parameterisation."""
    R = float(R)
    r = float(r)
    return r * (R + r * torch.cos(v))


def torus_laplace_beltrami(f, uv: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """Laplace–Beltrami of f ∘ X where X(u,v) embeds the torus."""
    if uv.shape[-1] != 2:
        raise ValueError("uv tensor must have last dimension 2")
    uv = uv.requires_grad_(True)
    u = uv[..., 0]
    v = uv[..., 1]
    x = torus_embed(u, v, R, r)
    y = f(x)
    if y.ndim > uv.ndim:
        y = y.squeeze(-1)
    ones = torch.ones_like(y)
    grad_uv = torch.autograd.grad(y, uv, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    du = grad_uv[..., 0]
    dv = grad_uv[..., 1]
    du2 = torch.autograd.grad(du, uv, grad_outputs=torch.ones_like(du), create_graph=True, retain_graph=True)[0][..., 0]
    dv_grads = torch.autograd.grad(dv, uv, grad_outputs=torch.ones_like(dv), create_graph=True)[0]
    dv2 = dv_grads[..., 1]
    R_term = R + r * torch.cos(v)
    lap = du2 / (R_term * R_term) + dv2 / (r * r) - torch.sin(v) / (r * R_term) * dv
    return lap.unsqueeze(-1)


def torus_normals(uv: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """Unit outward normals on the torus at parameters uv."""
    u = uv[..., 0]
    v = uv[..., 1]
    R = float(R)
    r = float(r)
    Xu = torch.stack([
        -(R + r * torch.cos(v)) * torch.sin(u),
        (R + r * torch.cos(v)) * torch.cos(u),
        torch.zeros_like(u)
    ], dim=-1)
    Xv = torch.stack([
        -r * torch.sin(v) * torch.cos(u),
        -r * torch.sin(v) * torch.sin(u),
        r * torch.cos(v)
    ], dim=-1)
    normals = torch.cross(Xu, Xv, dim=-1)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-12)
    return normals


def torus_value_and_grad(f, x: torch.Tensor):
    """Return value and ambient gradient of f at x ∈ R^3."""
    x = x.requires_grad_(True)
    y = f(x)
    if y.ndim > x.ndim:
        y = y.squeeze(-1)
    ones = torch.ones_like(y)
    grad = torch.autograd.grad(y, x, grad_outputs=ones, create_graph=True)[0]
    return y.unsqueeze(-1), grad
