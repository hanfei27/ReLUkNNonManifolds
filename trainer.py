import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
from utils.sampling import fibonacci_sphere, sample_torus
from equations.targets import (linear_target, quadratic_traceless_target,
                               l1_plus_l2, torus_fourier_target)
from utils.autodiff import laplace_beltrami
from equations.torus_ops import torus_laplace_beltrami, torus_normals
from equations.sphere_ops import proj_matrix
from models.relu_k_network import SphereNet


def build_target(name, geometry, cfg):
    if geometry == "sphere":
        if name == "linear":
            f, rhs = linear_target()
            return {"f": f, "laplace": None, "laplace_scale": None, "rhs": rhs}
        if name == "quad":
            f, rhs = quadratic_traceless_target()
            return {"f": f, "laplace": None, "laplace_scale": None, "rhs": rhs}
        if name == "l1_plus_l2":
            f, rhs = l1_plus_l2()
            return {"f": f, "laplace": None, "laplace_scale": None, "rhs": rhs}
        if isinstance(name, str) and name.startswith("sph_l_"):
            try:
                l = int(name.split("sph_l_")[-1])
                from equations.targets import spherical_harmonic_l

                f, rhs = spherical_harmonic_l(l)
                eigen_val = float(l * (l + int(cfg.get("ambient_dim", 3)) - 2))
                lap = lambda x: -eigen_val * f(x)
                return {"f": f, "laplace": lap, "laplace_scale": eigen_val, "rhs": rhs}
            except Exception:
                pass
        raise ValueError(f"unknown sphere target {name}")

    if geometry == "torus":
        R = float(cfg.get("torus_R", 2.0))
        r = float(cfg.get("torus_r", 0.5))
        m = int(cfg.get("torus_mode_u", 1))
        n = int(cfg.get("torus_mode_v", 1))
        amp_u = float(cfg.get("torus_amp_u", 1.0))
        amp_v = float(cfg.get("torus_amp_v", 0.5))
        if name in ("torus_fourier", "fourier"):
            f, lap = torus_fourier_target(R=R, r=r, m=m, n=n, amp_u=amp_u, amp_v=amp_v)
            return {"f": f, "laplace": lap, "laplace_scale": None, "rhs": None}
        raise ValueError(f"unknown torus target {name}")

    raise ValueError(f"unsupported geometry {geometry}")

def run_training(cfg, device):
    n = int(cfg.get("samples", 4096))
    s_order = int(cfg.get("s_order", 1))
    steps = int(cfg.get("steps", 3000))
    task = cfg.get("task","approx")
    target_name = cfg.get("target","l1_plus_l2")
    geometry = str(cfg.get("geometry", "sphere")).lower()
    if geometry not in ("sphere", "torus"):
        raise ValueError(f"geometry must be 'sphere' or 'torus', got {geometry}")

    ambient_dim = int(cfg.get("ambient_dim", 3)) if geometry == "sphere" else 3
    value_weight = float(cfg.get("value_weight", 1.0))
    laplace_weight = float(cfg.get("laplace_weight", 1.0))
    grad_weight = float(cfg.get("grad_weight", 0.0))

    if geometry == "sphere":
        x, w = fibonacci_sphere(n, device=device, dtype=torch.float32, ambient_dim=ambient_dim)
        aux = {}
    else:
        R = float(cfg.get("torus_R", 2.0))
        r = float(cfg.get("torus_r", 0.5))
        x, w, uv = sample_torus(n, device=device, dtype=torch.float32, R=R, r=r)
        aux = {"uv": uv.detach(), "R": R, "r": r}

    x = x.detach()
    w = w.detach()

    target_info = build_target(target_name, geometry, cfg)
    f_star = target_info["f"]
    rhs = target_info.get("rhs", None)
    laplace_star_fn = target_info.get("laplace", None)
    laplace_scale = target_info.get("laplace_scale", None)

    model = SphereNet(width=cfg.get("width",128),
                      depth=cfg.get("depth",4),
                      k=cfg.get("activation_k",3),
                      weight_clip=cfg.get("weight_clip",1.0),
                      normalize_activation=cfg.get("activation_normalize", True),
                      activation_target_var=cfg.get("activation_target_variance", 0.5),
                      use_layer_norm=cfg.get("activation_layer_norm", False),
                      activation_clamp=cfg.get("activation_clamp_max", None),
                      layer_widths=cfg.get("layer_widths", None),
                      activation_orders=cfg.get("layer_activation_k", cfg.get("layer_activation_orders", None)),
                      input_dim=ambient_dim,
                      normalize_input=(geometry == "sphere")).to(device)

    opt = optim.Adam(model.parameters(), lr=float(cfg.get("lr",1e-3)))
    grad_clip = cfg.get("grad_clip_norm", None)

    # Optional learning-rate schedule (configured via cfg['lr_schedule'])
    scheduler = None
    sched_cfg = cfg.get("lr_schedule", None)
    if sched_cfg:
        sched_type = str(sched_cfg.get("type", "step")).lower()
        if sched_type == "step":
            milestones = [int(m) for m in sched_cfg.get("milestones", [])]
            gamma = float(sched_cfg.get("gamma", 0.5))
            if milestones:
                scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
        elif sched_type in ("cosine", "cosineannealing"):
            t_max = int(sched_cfg.get("t_max", steps))
            eta_min = float(sched_cfg.get("eta_min", 0.0))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)
        elif sched_type == "exponential":
            gamma = float(sched_cfg.get("gamma", 0.5))
            scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    history = []
    # determine output directory: allow caller to pass a specific out_dir via cfg
    if 'out_dir' in cfg and cfg['out_dir']:
        out_dir = cfg['out_dir']
    else:
        # use timestamp-only directory for outputs (files will be inside results/<timestamp>/)
        ts = int(time.time())
        out_dir = os.path.join("results", f"{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # best model tracking
    best_loss = float('inf')
    best_step = -1
    save_best = bool(cfg.get('save_best', True))

    for step in range(1, steps+1):
        idx = torch.randint(0, n, (min(cfg.get("batch_size",1024), n),), device=device)
        xb = x[idx]
        wb = w[idx]
        uvb = aux.get("uv", None)
        if uvb is not None:
            uvb = uvb[idx]

        def f_theta(z): return model(z)
        if task=="approx":
            need_grad = (s_order >= 1) or grad_weight > 0.0
            need_lap = (s_order>=2)
            y = f_theta(xb)
            with torch.no_grad():
                y_star = f_star(xb)
            loss = value_weight * (wb * (y - y_star).squeeze(-1)**2).sum()
            if need_grad:
                def _grad(fn, pts):
                    with torch.enable_grad():
                        pts_req = pts.clone().detach().requires_grad_(True)
                        vals = fn(pts_req).squeeze(-1)
                        if not vals.requires_grad:
                            raise RuntimeError("target evaluation does not track gradients; check target definition")
                        ones = torch.ones_like(vals)
                        grads = torch.autograd.grad(vals, pts_req, grad_outputs=ones, create_graph=False)[0]
                    return grads

                g_theta = _grad(f_theta, xb)
                with torch.no_grad():
                    g_star = _grad(f_star, xb)
                if geometry == "torus":
                    normals = torus_normals(uvb, aux["R"], aux["r"])
                    g_theta = g_theta - (g_theta * normals).sum(dim=-1, keepdim=True) * normals
                    g_star = g_star - (g_star * normals).sum(dim=-1, keepdim=True) * normals
                elif geometry == "sphere":
                    P = proj_matrix(xb)
                    g_theta = torch.einsum('...ij,...j->...i', P, g_theta)
                    g_star = torch.einsum('...ij,...j->...i', P, g_star)
                grad_res = g_theta - g_star
                loss = loss + grad_weight * (wb * grad_res.pow(2).sum(dim=-1)).sum()
            if need_lap:
                if laplace_star_fn is not None:
                    with torch.no_grad():
                        lb_star = laplace_star_fn(xb)
                else:
                    lb_star = laplace_beltrami(f_star, xb)
                if geometry == "sphere":
                    lb = laplace_beltrami(f_theta, xb)
                else:
                    if uvb is None:
                        raise RuntimeError("torus Laplace requires uv coordinates")
                    lb = torus_laplace_beltrami(f_theta, uvb.detach(), aux["R"], aux["r"])
                lb_res = (lb - lb_star).squeeze(-1)
                if laplace_scale and laplace_scale > 0:
                    lb_res = lb_res / laplace_scale
                loss = loss + laplace_weight * (wb * lb_res ** 2).sum()
        elif task=="poisson":
            if geometry != "sphere":
                raise NotImplementedError("Poisson training currently only supports sphere geometry")
            # Poisson residual: Δ_S u = h; add mean-zero constraint
            y = f_theta(xb)
            lb = laplace_beltrami(f_theta, xb)
            with torch.no_grad():
                h = rhs(xb)
            res = (lb - h)
            loss = (wb*(res.squeeze(-1)**2)).sum()
            mean_pen = (y.mean())**2
            loss = loss + 1e-1*mean_pen
        else:
            raise ValueError

        opt.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn_utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        if scheduler is not None:
            scheduler.step()
        model.clip_weights_()

        if step==1 or step%int(cfg.get("log_interval",100))==0:
            cur_loss = float(loss.detach().cpu().item())
            history.append({"step": step, "loss": cur_loss})
            # save best model if improved
            if save_best and cur_loss < best_loss:
                best_loss = cur_loss
                best_step = step
                try:
                    torch.save({"model": model.state_dict(), "cfg": cfg, "meta": {"best_step": best_step, "best_loss": best_loss}}, os.path.join(out_dir, "model_best.pt"))
                except Exception:
                    pass

    meta = {"out_dir": out_dir,
            "task": task,
            "s_order": s_order,
            "target": target_name,
            "geometry": geometry,
            "best_step": best_step,
            "best_loss": best_loss}
    if geometry == "torus":
        meta["torus_R"] = aux["R"]
        meta["torus_r"] = aux["r"]
    torch.save({"model": model.state_dict(), "cfg": cfg, "meta": meta}, os.path.join(out_dir, "model_ckpt.pt"))
    return history, model, meta
