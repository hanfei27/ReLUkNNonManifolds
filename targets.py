import math
import torch
from equations.torus_ops import torus_angles


def _pad_vector(base: torch.Tensor, dim: int, device, dtype):
    vec = base.to(device=device, dtype=dtype).view(1, -1)
    cur = vec.shape[-1]
    if cur == dim:
        return vec
    if cur < dim:
        pad = torch.zeros(1, dim - cur, device=device, dtype=dtype)
        return torch.cat([vec, pad], dim=-1)
    return vec[..., :dim]


def _pad_matrix(base: torch.Tensor, dim: int, device, dtype):
    mat = base.to(device=device, dtype=dtype)
    cur = mat.shape[-1]
    if cur == dim:
        return mat
    if cur < dim:
        pad_cols = torch.zeros(cur, dim - cur, device=device, dtype=dtype)
        top = torch.cat([mat, pad_cols], dim=-1)
        pad_rows = torch.zeros(dim - cur, dim, device=device, dtype=dtype)
        return torch.cat([top, pad_rows], dim=0)
    return mat[:dim, :dim]

# Analytic targets built from low-degree spherical harmonics.
# For general S^{d-1}, zonal harmonics obey Δ_S Y_l = -l(l+d-2) Y_l.

def linear_target(a=None):
    # f*(x)= a·x
    if a is None:
        a = torch.tensor([1.0, -0.5, 0.75])
    a = a.view(1,3)
    def f(x):
        # ensure constant is on same device and dtype as input
        dim = x.shape[-1]
        aa = _pad_vector(a, dim, x.device, x.dtype)
        return (x * aa).sum(dim=-1, keepdim=True)
    def rhs_poisson(x):
        # Δ_S f = -2 f
        dim = x.shape[-1]
        return -(dim - 1.0) * f(x)
    return f, rhs_poisson

def quadratic_traceless_target(A=None):
    # f*(x)= x^T A x - (tr A)/3   with A symmetric
    if A is None:
        A = torch.tensor([[1.0, 0.2, -0.1],
                          [0.2, -0.5, 0.3],
                          [-0.1, 0.3, 0.2]])
    A = 0.5*(A+A.T)
    def f(x):
        # x: (...,3)
        dim = x.shape[-1]
        A_local = _pad_matrix(A, dim, x.device, x.dtype)
        Ax = torch.einsum("ij,...j->...i", A_local, x)
        trA_local = torch.trace(A_local)
        quad = (x*Ax).sum(dim=-1, keepdim=True) - (trA_local / dim)
        return quad
    def rhs_poisson(x):
        # For traceless part, Δ_S f = -6 f; the constant is harmonic => Δ_S const = 0
        dim = x.shape[-1]
        return -(2.0 * dim) * f(x)
    return f, rhs_poisson

def l1_plus_l2():
    f1,_ = linear_target()
    f2,_ = quadratic_traceless_target()
    def f(x):
        return f1(x)+0.5*f2(x)
    def rhs_poisson(x):
        dim = x.shape[-1]
        return -(dim - 1.0) * f1(x) - (dim) * f2(x)  # eigenvalues l=1 and l=2 scaled appropriately
    return f, rhs_poisson

def single_sph_l1(k=0):
    """
    Single spherical harmonic of degree l=1: choose component k in {0,1,2} corresponding to x,y,z.
    Y(x) = x_k, and Δ_S Y = -2 Y on S^2.
    Returns (f, rhs) where rhs(x) = -2 f(x).
    """
    assert k in (0,1,2)
    def f(x):
        # x: (...,>=3)
        return x[..., k:k+1]
    def rhs_poisson(x):
        dim = x.shape[-1]
        return -(dim - 1.0) * f(x)
    return f, rhs_poisson

def spherical_harmonic_l(l: int):
    """Unnormalised zonal spherical harmonic on S^{d-1} depending on the last axis."""
    assert isinstance(l, int) and l >= 0

    def gegenbauer_poly(z, degree, alpha):
        if degree == 0:
            return torch.ones_like(z)
        if degree == 1:
            return (2.0 * alpha) * z
        C0 = torch.ones_like(z)
        C1 = (2.0 * alpha) * z
        for n in range(2, degree + 1):
            a = 2.0 * (n + alpha - 1.0) / n
            b = (n + 2.0 * alpha - 2.0) / n
            Cn = a * z * C1 - b * C0
            C0, C1 = C1, Cn
        return C1

    def f(x):
        dim = x.shape[-1]
        if dim < 2:
            raise ValueError("ambient dimension must be >= 2 for spherical harmonics")
        z = x[..., -1]
        if dim == 2:
            theta = torch.acos(torch.clamp(z, -1.0, 1.0))
            C = torch.cos(l * theta)
        else:
            alpha = 0.5 * (dim - 2.0)
            C = gegenbauer_poly(z, l, alpha)
        sphere_area = 2.0 * math.pi ** (dim / 2.0) / math.gamma(dim / 2.0)
        if dim == 2:
            integral_total = 2.0 * math.pi if l == 0 else math.pi
        else:
            alpha = 0.5 * (dim - 2.0)
            area_sub = 2.0 * math.pi ** ((dim - 1.0) / 2.0) / math.gamma((dim - 1.0) / 2.0)
            integral_t = (math.pi * (2.0 ** (1.0 - 2.0 * alpha)) * math.gamma(l + 2.0 * alpha)
                          ) / (math.factorial(l) * (l + alpha) * (math.gamma(alpha) ** 2))
            integral_total = area_sub * integral_t
            if l == 0:
                integral_total = sphere_area
        scale = math.sqrt(sphere_area / integral_total) if integral_total > 0 else 1.0
        return (scale * C).unsqueeze(-1)

    def rhs_poisson(x):
        dim = x.shape[-1]
        return -float(l * (l + dim - 2)) * f(x)

    return f, rhs_poisson


def torus_fourier_target(R: float = 2.0, r: float = 0.5,
                         m: int = 1, n: int = 1,
                         amp_u: float = 1.0, amp_v: float = 0.5):
    """Analytic target on a standard torus with explicit Laplace–Beltrami."""
    R = float(R)
    r = float(r)
    m = int(m)
    n = int(n)
    A = float(amp_u)
    B = float(amp_v)

    if r <= 0 or R <= r:
        raise ValueError("Require major radius R > minor radius r > 0")
    if m < 0 or n < 0:
        raise ValueError("Fourier modes m, n must be non-negative")

    def _angles(x):
        return torus_angles(x, R, r).unbind(-1)

    def f(x):
        u, v = _angles(x)
        val = A * torch.cos(m * u)
        if B != 0.0 and n > 0:
            val = val + B * torch.sin(n * v)
        return val.unsqueeze(-1)

    def laplace(x):
        u, v = _angles(x)
        R_term = R + r * torch.cos(v)
        lap_u = torch.zeros_like(u)
        lap_v = torch.zeros_like(v)
        if A != 0.0 and m > 0:
            lap_u = -A * (m ** 2) * torch.cos(m * u) / (R_term * R_term)
        if B != 0.0 and n > 0:
            lap_v = (-B * (n ** 2) * torch.sin(n * v) / (r * r)
                     - B * n * torch.cos(n * v) * torch.sin(v) / (r * R_term))
        return (lap_u + lap_v).unsqueeze(-1)

    return f, laplace
