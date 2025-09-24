
import torch, math
from equations.torus_ops import torus_embed, torus_area_element, torus_surface_area


def _sphere_surface_area(dim: int) -> float:
    """Surface area of S^{dim-1} in R^{dim}."""
    if dim < 1:
        raise ValueError('dimension must be >= 1')
    return 2.0 * math.pi ** (dim / 2.0) / math.gamma(dim / 2.0)


def fibonacci_sphere(n: int, device=None, dtype=None, ambient_dim: int = 3):
    """Generate quasi-uniform samples on S^{ambient_dim-1} ⊂ R^{ambient_dim}."""
    device = device or "cpu"
    dtype = dtype or torch.float32
    n = int(n)
    ambient_dim = int(ambient_dim)
    if ambient_dim < 2:
        raise ValueError("ambient_dim must be at least 2 for a sphere")

    if ambient_dim == 3:
        k = torch.arange(n, device=device, dtype=dtype)
        phi = (1.0 + 5.0 ** 0.5) / 2.0
        ga = 2.0 * math.pi * (1.0 - 1.0 / phi)
        z = 1.0 - 2.0 * (k + 0.5) / n
        r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        theta = ga * (k + 0.5)
        pts = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)
    else:
        # use Sobol low-discrepancy points pushed through Gaussian -> normalised
        try:
            from torch.quasirandom import SobolEngine

            sobol = SobolEngine(dimension=ambient_dim, scramble=True, seed=torch.initial_seed() & 0xFFFFFFFF)
            u = sobol.draw(n).to(dtype=torch.float64)  # high precision before erfinv
            eps = 1e-7
            u = u.clamp(min=eps, max=1.0 - eps)
            g = torch.special.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)
            pts = g / (g.norm(dim=-1, keepdim=True) + 1e-12)
            pts = pts.to(device=device, dtype=dtype)
        except Exception:
            g = torch.randn(n, ambient_dim, device=device, dtype=dtype)
            pts = g / (g.norm(dim=-1, keepdim=True) + 1e-12)

    area = _sphere_surface_area(ambient_dim)
    w = area / n * torch.ones(n, device=device, dtype=dtype)
    return pts, w


def sample_torus(n: int, device=None, dtype=None, R: float = 2.0, r: float = 0.5):
    """Generate Monte Carlo samples on a standard torus embedded in R^3."""
    device = device or "cpu"
    dtype = dtype or torch.float32
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    major = float(R)
    minor = float(r)
    if minor <= 0 or major <= minor:
        raise ValueError("Require major radius R > minor radius r > 0 for a valid torus")
    two_pi = 2.0 * math.pi
    u = two_pi * torch.rand(n, device=device, dtype=dtype)
    v = two_pi * torch.rand(n, device=device, dtype=dtype)
    pts = torus_embed(u, v, major, minor)
    uv = torch.stack([u, v], dim=-1)
    jac = torus_area_element(v, major, minor)
    area = torus_surface_area(major, minor)
    weights = (area / n) * (jac / (major * minor))
    return pts.to(dtype=dtype), weights.to(dtype=dtype), uv
