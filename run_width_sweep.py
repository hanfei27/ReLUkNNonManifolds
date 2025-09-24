"""Width scaling sweep with repeats and Sobolev proxy metrics.

This script runs a sweep over model widths and records the residual proxies
for ``H_s`` norms (value, tangential gradient, Laplace–Beltrami) as suggested
in the simultaneous approximation theory discussion. Each width can be run
multiple times with different random seeds; statistics are aggregated and
written to ``results/width_sweep_<ts>/`` along with diagnostic plots against the
integer width index and the total parameter count.
"""
import argparse
import csv
import math
import os
import random
import statistics
import time

import torch
import yaml

from train.trainer import build_target, run_training
from utils.autodiff import laplace_beltrami
from utils.sampling import fibonacci_sphere, sample_torus
from equations.torus_ops import torus_laplace_beltrami, torus_normals
from equations.sphere_ops import proj_matrix

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Configuration defaults
# Prefer the width-specific config as the default base when present
DEFAULT_BASE_CONFIG = "config_width.yaml"
DEFAULT_WIDTHS = [64, 128, 256]
DEFAULT_STEPS = 300
DEFAULT_SAMPLES = 2048
DEFAULT_BATCH = 512
DEFAULT_REPEATS = 1
DEFAULT_WIDTH_CONFIG = "config_width.yaml"
DEFAULT_EVAL_SAMPLES = 2048


def parse_widths_arg(arg):
    if arg is None:
        return None
    s = str(arg)
    if ',' in s:
        return [int(x) for x in s.split(',') if x.strip()]
    return [int(s)]


def load_base_cfg(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_random_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except Exception:
        pass


def total_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def mse_integral(residual: torch.Tensor, weights: torch.Tensor) -> float:
    r"""Return weighted mean of squared residuals."""
    val = (weights * residual.squeeze(-1) ** 2).sum()
    den = weights.sum() + 1e-12
    return float((val / den).detach().cpu().item())


def grad_mse(g_diff: torch.Tensor, weights: torch.Tensor) -> float:
    r"""Return weighted mean of squared gradient residuals (gradient MSE)."""
    val = (weights * g_diff.pow(2).sum(dim=-1)).sum()
    den = weights.sum() + 1e-12
    return float((val / den).detach().cpu().item())


def run_one(cfg, device):
    history, model, meta = run_training(cfg, device)
    final_loss = history[-1]['loss'] if len(history) else float('nan')
    return final_loss, history, model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--widths', type=str, default=None,
                    help="model widths, e.g. '64,128,256' or '128'")
    ap.add_argument('--width-config', type=str, default=None,
                    help='YAML file that specifies widths (fields: widths or widths_list)')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--samples', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--base-config', type=str, default=DEFAULT_BASE_CONFIG)
    ap.add_argument('--device', type=str, default=None, help='override device in base config')
    ap.add_argument('--repeats', type=int, default=None, help='number of seeds per width')
    ap.add_argument('--ambient-dim', type=int, default=None,
                    help='ambient dimension to embed the 2-sphere (>=3)')
    ap.add_argument('--dry-run', action='store_true', help='only print planned runs without training')
    args = ap.parse_args()

    if args.width_config is None and os.path.exists(DEFAULT_WIDTH_CONFIG):
        args.width_config = DEFAULT_WIDTH_CONFIG

    if args.base_config == DEFAULT_BASE_CONFIG and args.width_config == DEFAULT_WIDTH_CONFIG:
        if not os.path.exists(args.base_config) and os.path.exists(args.width_config):
            args.base_config = args.width_config

    base = load_base_cfg(args.base_config)

    widths = parse_widths_arg(args.widths) or None
    if widths is None and args.width_config:
        try:
            with open(args.width_config, 'r', encoding='utf-8') as f:
                wcfg = yaml.safe_load(f)
            if isinstance(wcfg, dict):
                if 'widths_list' in wcfg and wcfg['widths_list'] is not None:
                    widths = [int(x) for x in wcfg['widths_list']]
                elif 'widths' in wcfg and wcfg['widths'] is not None:
                    raw = wcfg['widths']
                    if isinstance(raw, list):
                        widths = [int(x) for x in raw]
                    else:
                        widths = parse_widths_arg(str(raw))
        except Exception as e:
            print(f"[WARN] failed to read width config {args.width_config}: {e}")
    widths = widths or DEFAULT_WIDTHS

    steps = args.steps if args.steps is not None else int(base.get('steps', DEFAULT_STEPS))
    samples = args.samples if args.samples is not None else int(base.get('samples', DEFAULT_SAMPLES))
    batch = args.batch_size if args.batch_size is not None else int(base.get('batch_size', DEFAULT_BATCH))
    repeats = args.repeats if args.repeats is not None else int(base.get('repeats', DEFAULT_REPEATS))
    eval_samples = int(base.get('eval_samples', DEFAULT_EVAL_SAMPLES))
    ambient_dim = args.ambient_dim if args.ambient_dim is not None else int(base.get('ambient_dim', 3))
    geometry = str(base.get('geometry', 'sphere')).lower()
    if geometry not in ('sphere', 'torus'):
        raise ValueError(f"Unsupported geometry '{geometry}' in base config")

    base['steps'] = steps
    base['samples'] = samples
    base['batch_size'] = batch
    base['ambient_dim'] = ambient_dim
    base_seed = int(base.get('seed', 123))
    if args.device:
        base['device'] = args.device

    req = str(base.get('device', 'cpu'))
    req_low = req.lower()
    device = torch.device('cpu')
    # Prefer CUDA if requested and available (support 'cuda' or 'cuda:0' syntax)
    if req.startswith('cuda'):
        try:
            if torch.cuda.is_available():
                # allow specifying device index like 'cuda:0'
                try:
                    device = torch.device(req)
                except Exception:
                    device = torch.device('cuda')
            else:
                print('[WARN] CUDA requested but not available; falling back to CPU')
                device = torch.device('cpu')
        except Exception:
            device = torch.device('cpu')
    elif req.startswith('mps'):
        try:
            if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                device = torch.device(req)
            else:
                print('[WARN] MPS requested but not available; falling back to CPU')
                device = torch.device('cpu')
        except Exception:
            device = torch.device('cpu')

    # diagnostic prints: show requested device, chosen device and CUDA availability
    try:
        print(f"Requested device (from config/args): {req}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}")
    except Exception:
        pass

    ts = int(time.time())
    out_dir = os.path.join('results', f'width_sweep_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'results.csv')
    rows = []
    raw_rows = []

    print(f"Planned widths: {widths}")
    print(f"Using device: {device}")
    print(f"Steps={steps}, samples={samples}, batch_size={batch}, repeats={repeats}")
    print(f"Ambient dimension={ambient_dim}")
    print(f"Geometry={geometry}")
    if args.dry_run:
        print('Dry run; exiting without training')
        sanitized_base = dict(base)
        sanitized_base.pop('width', None)
        sweep_cfg = {
            'timestamp': ts,
            'width_list': widths,
            'steps': steps,
            'samples': samples,
            'batch_size': batch,
            'device': str(device),
            'base_config': sanitized_base,
            'repeats': repeats,
        }
        with open(os.path.join(out_dir, 'sweep_config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(sweep_cfg, f, sort_keys=False, allow_unicode=True)
        return

    fieldnames_raw = ['width', 'repeat', 'seed', 'param_count', 'final_loss',
                      'value_mse', 'grad_mse', 'laplace_mse']
    fieldnames_agg = ['width', 'param_count',
                      'value_mse_mean', 'value_mse_std',
                      'grad_mse_mean', 'grad_mse_std',
                      'laplace_mse_mean', 'laplace_mse_std',
                      'final_loss_mean', 'final_loss_std']

    for w in widths:
        width = int(w)
        per_w_dir = os.path.join(out_dir, f'w_{width}')
        os.makedirs(per_w_dir, exist_ok=True)

        per_width_runs = []
        for rep in range(repeats):
            cfg = dict(base)
            cfg['width'] = width
            cfg['out_dir'] = os.path.join(per_w_dir, f'rep_{rep}')
            os.makedirs(cfg['out_dir'], exist_ok=True)

            seed = base_seed + rep + width * 1000
            cfg['seed'] = seed
            set_random_seeds(seed)

            print(f"Running width={width}, repeat={rep+1}/{repeats} (seed={seed}) on device={device} ...")
            final_loss, history, model, meta = run_one(cfg, device)

            param_count = total_parameters(model)

            if geometry == 'sphere':
                x_eval, w_eval = fibonacci_sphere(eval_samples, device=device, dtype=torch.float32,
                                                  ambient_dim=ambient_dim)
                aux_uv = None
            else:
                R = float(cfg.get('torus_R', 1.5))
                r_minor = float(cfg.get('torus_r', 0.5))
                x_eval, w_eval, uv_eval = sample_torus(eval_samples, device=device, dtype=torch.float32,
                                                       R=R, r=r_minor)
                aux_uv = uv_eval
            w_eval = w_eval.detach()
            x_eval = x_eval.detach()

            target_name = cfg.get('target', None)
            target_info = build_target(target_name, geometry, cfg)
            f_star = target_info['f']
            laplace_star_fn = target_info.get('laplace')

            def f_theta(z):
                return model(z)

            def value_and_grad(fn, pts):
                pts_req = pts.clone().detach().requires_grad_(True)
                vals = fn(pts_req)
                vals_s = vals.squeeze(-1)
                ones = torch.ones_like(vals_s)
                grads = torch.autograd.grad(vals_s, pts_req, grad_outputs=ones, create_graph=False)[0]
                return vals.detach(), grads.detach()

            y_pred, g_pred = value_and_grad(f_theta, x_eval)
            y_star, g_star = value_and_grad(f_star, x_eval)
            if geometry == 'torus':
                normals_eval = torus_normals(aux_uv, R, r_minor)
                g_pred = g_pred - (g_pred * normals_eval).sum(dim=-1, keepdim=True) * normals_eval
                g_star = g_star - (g_star * normals_eval).sum(dim=-1, keepdim=True) * normals_eval

            if geometry == 'sphere':
                lb_pred = laplace_beltrami(f_theta, x_eval)
                if laplace_star_fn is not None:
                    with torch.no_grad():
                        lb_star = laplace_star_fn(x_eval)
                else:
                    lb_star = laplace_beltrami(f_star, x_eval)
            else:
                R = float(cfg.get('torus_R', 1.5))
                r_minor = float(cfg.get('torus_r', 0.5))
                if aux_uv is None:
                    raise RuntimeError('Tor us geometry requires uv coordinates for Laplace evaluation')
                lb_pred = torus_laplace_beltrami(f_theta, aux_uv, R, r_minor)
                if laplace_star_fn is not None:
                    with torch.no_grad():
                        lb_star = laplace_star_fn(x_eval)
                else:
                    lb_star = torus_laplace_beltrami(f_star, aux_uv, R, r_minor)

            if geometry == 'sphere':
                P_eval = proj_matrix(x_eval)
                g_pred = torch.einsum('...ij,...j->...i', P_eval, g_pred)
                g_star = torch.einsum('...ij,...j->...i', P_eval, g_star)
            residual_val = (y_pred - y_star)
            residual_lap = (lb_pred - lb_star)
            grad_diff = (g_pred - g_star)

            value_mse = mse_integral(residual_val, w_eval)
            grad_mse_val = grad_mse(grad_diff, w_eval)
            laplace_mse = mse_integral(residual_lap, w_eval)

            per_width_runs.append({
                'width': width,
                'repeat': rep,
                'seed': seed,
                'param_count': param_count,
                'final_loss': final_loss,
                'value_mse': value_mse,
                'grad_mse': grad_mse_val,
                'laplace_mse': laplace_mse,
            })

            raw_rows.append(per_width_runs[-1])

        # aggregate statistics for this width
        def _collect(key):
            vals = [run[key] for run in per_width_runs if isinstance(run[key], float) and math.isfinite(run[key])]
            return vals

        def _mean_std(values):
            if not values:
                return float('nan'), float('nan')
            if len(values) == 1:
                return float(values[0]), 0.0
            try:
                return float(statistics.mean(values)), float(statistics.stdev(values))
            except statistics.StatisticsError:
                return float(values[0]), 0.0

        value_mean, value_std = _mean_std(_collect('value_mse'))
        grad_mean, grad_std = _mean_std(_collect('grad_mse'))
        lap_mean, lap_std = _mean_std(_collect('laplace_mse'))
        loss_mean, loss_std = _mean_std(_collect('final_loss'))

        rows.append({
            'width': width,
            'param_count': per_width_runs[0]['param_count'] if per_width_runs else float('nan'),
            'value_mse_mean': value_mean,
            'value_mse_std': value_std,
            'grad_mse_mean': grad_mean,
            'grad_mse_std': grad_std,
            'laplace_mse_mean': lap_mean,
            'laplace_mse_std': lap_std,
            'final_loss_mean': loss_mean,
            'final_loss_std': loss_std,
        })

    rows.sort(key=lambda r: r['width'])
    raw_rows.sort(key=lambda r: (r['width'], r['repeat']))

    # write raw and aggregated CSVs
    raw_csv_path = os.path.join(out_dir, 'results_raw.csv')
    with open(raw_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_raw)
        writer.writeheader()
        for r in raw_rows:
            writer.writerow(r)

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_agg)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    widths_plot = [r['width'] for r in rows]
    width_ticks = sorted({int(w) for w in widths_plot if isinstance(w, (int, float)) and math.isfinite(w)})
    param_counts = [r['param_count'] for r in rows]

    # final loss: plot log10(mean) with asymmetric errorbars derived from mean +/- std
    def _log10_and_err(mean_list, std_list, eps=1e-12):
        ys = []
        lower_err = []
        upper_err = []
        for m, s in zip(mean_list, std_list):
            try:
                m_val = float(m) if (m is not None and math.isfinite(m)) else 0.0
                s_val = float(s) if (s is not None and math.isfinite(s)) else 0.0
                m_safe = max(m_val, eps)
                low = max(m_safe - s_val, eps)
                high = max(m_safe + s_val, eps)
                y = math.log10(m_safe)
                lower_err.append(y - math.log10(low))
                upper_err.append(math.log10(high) - y)
                ys.append(y)
            except Exception:
                ys.append(math.log10(eps))
                lower_err.append(0.0)
                upper_err.append(0.0)
        return ys, [lower_err, upper_err]

    final_losses = [r.get('final_loss_mean', float('nan')) for r in rows]
    final_std = [r.get('final_loss_std', 0.0) for r in rows]
    final_y, final_err = _log10_and_err(final_losses, final_std)

    plt.figure()
    ln_loss, = plt.plot(widths_plot, final_y, '-o', label='final loss ($\\log_{10}$)')
    color_loss = ln_loss.get_color()
    # shaded error band for final loss (asymmetric)
    lower_loss = [y - le for y, le in zip(final_y, final_err[0])]
    upper_loss = [y + ue for y, ue in zip(final_y, final_err[1])]
    plt.fill_between(widths_plot, lower_loss, upper_loss, color=color_loss, alpha=0.22)
    plt.errorbar(widths_plot, final_y, yerr=final_err, fmt='-', color=color_loss, capsize=3, elinewidth=1.0)
    plt.xlabel('width')
    if width_ticks:
        plt.xticks(width_ticks)
    plt.ylabel('$\\log_{10}(\\text{final loss})$')
    plt.title('Training loss vs width')
    plt.grid(True, which='both', ls='--')
    plt.savefig(os.path.join(out_dir, 'plot_loss.png'))

    # aggregated MSEs: plot log10 of means with asymmetric errorbars derived from mean +/- std
    value_ms = [r.get('value_mse_mean', float('nan')) for r in rows]
    value_std = [r.get('value_mse_std', 0.0) for r in rows]
    grad_ms = [r.get('grad_mse_mean', float('nan')) for r in rows]
    grad_std = [r.get('grad_mse_std', 0.0) for r in rows]
    lap_ms = [r.get('laplace_mse_mean', float('nan')) for r in rows]
    lap_std = [r.get('laplace_mse_std', 0.0) for r in rows]

    val_y, val_err = _log10_and_err(value_ms, value_std)
    grad_y, grad_err = _log10_and_err(grad_ms, grad_std)
    lap_y, lap_err = _log10_and_err(lap_ms, lap_std)

    plt.figure()
    widths_log2 = [math.log2(w) if (w is not None and w > 0 and math.isfinite(w)) else float('nan') for w in widths_plot]
    # shaded error bands and errorbars (log10 values)
    # val
    ln_val, = plt.plot(widths_log2, val_y, '-', label='value MSE ($\\log_{10}$)')
    color_val = ln_val.get_color()
    lower = [y - le for y, le in zip(val_y, val_err[0])]
    upper = [y + ue for y, ue in zip(val_y, val_err[1])]
    plt.fill_between(widths_log2, lower, upper, color=color_val, alpha=0.22)
    plt.errorbar(widths_log2, val_y, yerr=val_err, fmt='-', color=color_val, capsize=3, elinewidth=1.0)

    # grad
    ln_grad, = plt.plot(widths_log2, grad_y, '-', label='grad MSE ($\\log_{10}$)')
    color_grad = ln_grad.get_color()
    lower = [y - le for y, le in zip(grad_y, grad_err[0])]
    upper = [y + ue for y, ue in zip(grad_y, grad_err[1])]
    plt.fill_between(widths_log2, lower, upper, color=color_grad, alpha=0.22)
    plt.errorbar(widths_log2, grad_y, yerr=grad_err, fmt='-', color=color_grad, capsize=3, elinewidth=1.0)

    # lap
    ln_lap, = plt.plot(widths_log2, lap_y, '-', label='laplace MSE ($\\log_{10}$)')
    color_lap = ln_lap.get_color()
    lower = [y - le for y, le in zip(lap_y, lap_err[0])]
    upper = [y + ue for y, ue in zip(lap_y, lap_err[1])]
    plt.fill_between(widths_log2, lower, upper, color=color_lap, alpha=0.22)
    plt.errorbar(widths_log2, lap_y, yerr=lap_err, fmt='-', color=color_lap, capsize=3, elinewidth=1.0)

    plt.xlabel('$\\log_{2}(\\text{width})$')
    if width_ticks:
        xticks_log2 = [math.log2(w) for w in width_ticks]
        plt.xticks(xticks_log2, [f"{t:.1f}" for t in xticks_log2])
    plt.ylabel('$\\log_{10}(\\mathrm{MSE})$')
    plt.title('Value / Grad / Laplace MSE across $\\log_{2}(\\text{width})$')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.savefig(os.path.join(out_dir, 'plot_errors.png'))

    # plot against total parameter count (log10) for clearer scaling visualization
    plt.figure()
    param_counts_log10 = [math.log10(p) if (p is not None and p > 0 and math.isfinite(p)) else float('nan')
                          for p in param_counts]

    # repeat with clearer bands using log10(param count) on the x-axis
    ln_vp, = plt.plot(param_counts_log10, val_y, '-', label='value MSE ($\\log_{10}$)')
    color_vp = ln_vp.get_color()
    plt.fill_between(param_counts_log10, [y - le for y, le in zip(val_y, val_err[0])],
                     [y + ue for y, ue in zip(val_y, val_err[1])], color=color_vp, alpha=0.18)
    ln_gp, = plt.plot(param_counts_log10, grad_y, '-', label='grad MSE ($\\log_{10}$)')
    color_gp = ln_gp.get_color()
    plt.fill_between(param_counts_log10, [y - le for y, le in zip(grad_y, grad_err[0])],
                     [y + ue for y, ue in zip(grad_y, grad_err[1])], color=color_gp, alpha=0.18)
    ln_lp, = plt.plot(param_counts_log10, lap_y, '-', label='laplace MSE ($\\log_{10}$)')
    color_lp = ln_lp.get_color()
    plt.fill_between(param_counts_log10, [y - le for y, le in zip(lap_y, lap_err[0])],
                     [y + ue for y, ue in zip(lap_y, lap_err[1])], color=color_lp, alpha=0.18)
    plt.xlabel('$\\log_{10}(\\text{param count})$')
    plt.ylabel('$\\log_{10}(\\mathrm{MSE})$')
    plt.title('Value / Grad / Laplace MSE across $\\log_{10}(\\text{param count})$')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.savefig(os.path.join(out_dir, 'plot_errors_vs_params.png'))

    def _fit_line(xs, ys):
        pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys)
               if x is not None and y is not None and x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)]
        if len(pts) < 2:
            return float('nan'), float('nan')
        xs_log = [p[0] for p in pts]
        ys_log = [p[1] for p in pts]
        x_mean = sum(xs_log) / len(xs_log)
        y_mean = sum(ys_log) / len(ys_log)
        denom = sum((x - x_mean) ** 2 for x in xs_log)
        if denom == 0:
            return float('nan'), float('nan')
        numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs_log, ys_log))
        slope = numer / denom
        intercept = y_mean - slope * x_mean
        return slope, intercept

    slope_value, _ = _fit_line(param_counts, value_ms)
    slope_grad, _ = _fit_line(param_counts, grad_ms)
    slope_lap, _ = _fit_line(param_counts, lap_ms)
    # training loss slope (final_loss_mean vs parameter count)
    final_losses = [r.get('final_loss_mean', float('nan')) for r in rows]
    slope_loss, _ = _fit_line(param_counts, final_losses)

    slopes_txt = os.path.join(out_dir, 'scaling_slopes.txt')
    with open(slopes_txt, 'w', encoding='utf-8') as f:
        f.write('log-log slope estimates (log(E) vs log(S))\n')
        f.write(f'value ≈ {slope_value:.4f}\n')
        f.write(f'grad  ≈ {slope_grad:.4f}\n')
        f.write(f'lap   ≈ {slope_lap:.4f}\n')
        f.write(f'loss  ≈ {slope_loss:.4f}\n')
    print('[INFO] log-log slopes:', {'value': slope_value, 'grad': slope_grad, 'lap': slope_lap, 'loss': slope_loss})

    sanitized_base = dict(base)
    sanitized_base.pop('width', None)
    sweep_cfg = {
        'timestamp': ts,
        'width_list': widths,
        'steps': steps,
        'samples': samples,
        'batch_size': batch,
        'device': str(device),
        'repeats': repeats,
        'eval_samples': eval_samples,
        'ambient_dim': ambient_dim,
        'base_config': sanitized_base,
    }
    try:
        with open(os.path.join(out_dir, 'sweep_config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(sweep_cfg, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        print(f"[WARN] failed to write sweep_config.yaml: {e}")

    print('Sweep done ->', out_dir)


if __name__ == '__main__':
    main()
