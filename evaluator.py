import os
import torch
from utils.sampling import fibonacci_sphere, sample_torus
from utils.autodiff import laplace_beltrami
from utils.metrics import weighted_mse
from equations.torus_ops import torus_laplace_beltrami, torus_normals
from equations.sphere_ops import proj_matrix
from train.trainer import build_target


def _value_and_grad(fn, pts):
    pts_req = pts.clone().detach().requires_grad_(True)
    vals = fn(pts_req)
    vals_s = vals.squeeze(-1)
    ones = torch.ones_like(vals_s)
    grads = torch.autograd.grad(vals_s, pts_req, grad_outputs=ones, create_graph=False)[0]
    return vals.detach(), grads.detach()


def evaluate_model(cfg, device, model, meta):
    ntest = int(cfg.get("samples", 4096))
    geometry = str(meta.get("geometry", cfg.get("geometry", "sphere"))).lower()
    if geometry not in ("sphere", "torus"):
        raise ValueError(f"unsupported geometry {geometry}")

    ambient_dim = int(cfg.get("ambient_dim", 3)) if geometry == "sphere" else 3

    if geometry == "sphere":
        x, w = fibonacci_sphere(ntest, device=device, ambient_dim=ambient_dim)
        aux = {}
    else:
        R = float(meta.get("torus_R", cfg.get("torus_R", 2.0)))
        r = float(meta.get("torus_r", cfg.get("torus_r", 0.5)))
        x, w, uv = sample_torus(ntest, device=device, R=R, r=r)
        aux = {"uv": uv, "R": R, "r": r}

    cfg_local = dict(cfg)
    if geometry == "torus":
        cfg_local["torus_R"] = aux["R"]
        cfg_local["torus_r"] = aux["r"]

    target_info = build_target(meta["target"], geometry, cfg_local)
    f_star = target_info["f"]
    laplace_star_fn = target_info.get("laplace")

    def f_theta(z):
        return model(z)

    y, g = _value_and_grad(f_theta, x)
    y_star, g_star = _value_and_grad(f_star, x)
    if geometry == "torus":
        normals = torus_normals(aux["uv"], aux["R"], aux["r"])
        g = g - (g * normals).sum(dim=-1, keepdim=True) * normals
        g_star = g_star - (g_star * normals).sum(dim=-1, keepdim=True) * normals
    elif geometry == "sphere":
        P = proj_matrix(x)
        g = torch.einsum('...ij,...j->...i', P, g)
        g_star = torch.einsum('...ij,...j->...i', P, g_star)

    val_mse = weighted_mse(y - y_star, w)
    grad_err = (g - g_star).norm(dim=-1, keepdim=True)
    grad_mse = weighted_mse(grad_err, w)

    if geometry == "sphere":
        lb = laplace_beltrami(f_theta, x)
        if laplace_star_fn is not None:
            with torch.no_grad():
                lb_star = laplace_star_fn(x)
        else:
            lb_star = laplace_beltrami(f_star, x)
    else:
        lb = torus_laplace_beltrami(f_theta, aux["uv"], aux["R"], aux["r"])
        if laplace_star_fn is not None:
            with torch.no_grad():
                lb_star = laplace_star_fn(x)
        else:
            lb_star = torus_laplace_beltrami(f_star, aux["uv"], aux["R"], aux["r"])

    lb_mse = weighted_mse(lb - lb_star, w)

    out_dir = meta.get("out_dir", "results")
    out_path = os.path.join(out_dir, "report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {meta['task']}\n")
        f.write(f"Geometry: {geometry}\n")
        f.write(f"Target: {meta['target']}\n")
        f.write(f"Weighted MSE (value): {val_mse:.4e}\n")
        f.write(f"Weighted MSE (grad):  {grad_mse:.4e}\n")
        f.write(f"Weighted MSE (Laplace):  {lb_mse:.4e}\n")
    print("[EVAL]", out_path)
