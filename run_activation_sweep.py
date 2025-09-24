"""
Run a sweep over activation_k (ReLU^{k-1}) and record final loss.
Produces results/activation_sweep_<ts>/results.csv and plot.png

This script is standalone and uses the existing train.run_training function.
"""
import time, os, yaml, math, argparse, csv, random, statistics
from train.trainer import run_training, build_target
from test.evaluator import evaluate_model
from utils.sampling import fibonacci_sphere, sample_torus
from utils.autodiff import laplace_beltrami
from equations.torus_ops import torus_laplace_beltrami, torus_normals
from equations.sphere_ops import proj_matrix
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Configuration defaults
DEFAULT_BASE_CONFIG = "config.yaml"
DEFAULT_KS = list(range(2, 9))
DEFAULT_STEPS = 300
DEFAULT_SAMPLES = 2048
DEFAULT_BATCH = 512
DEFAULT_ACTIVATION_CONFIG = 'config_activation.yaml'


def parse_ks_arg(arg):
    # accept formats: "2,3,4" or "2-8"
    if arg is None:
        return None
    arg = str(arg)
    if ',' in arg:
        return [int(x) for x in arg.split(',') if x.strip()]
    if '-' in arg:
        a, b = arg.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(arg)]


def load_base_cfg(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_one(cfg, device):
    history, model, meta = run_training(cfg, device)
    final_loss = history[-1]['loss'] if len(history) else float('nan')
    return final_loss, history, model, meta


def set_random_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass


def _mean_std(values):
    if not values:
        return float('nan'), float('nan')
    if len(values) == 1:
        return float(values[0]), 0.0
    try:
        return float(statistics.mean(values)), float(statistics.stdev(values))
    except statistics.StatisticsError:
        return float(values[0]), 0.0


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


def mse_integral(residual: torch.Tensor, weights: torch.Tensor) -> float:
    val = (weights * residual.squeeze(-1) ** 2).sum()
    den = weights.sum() + 1e-12
    return float((val / den).detach().cpu().item())


def grad_mse_metric(residual_grad: torch.Tensor, weights: torch.Tensor) -> float:
    val = (weights * residual_grad.pow(2).sum(dim=-1)).sum()
    den = weights.sum() + 1e-12
    return float((val / den).detach().cpu().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', type=str, default=None,
                    help="activation ks, e.g. '2,3,4' or '2-8' or '5'")
    ap.add_argument('--activation-config', type=str, default=None,
                    help='YAML file that specifies activation ks (fields: ks or ks_list)')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--samples', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--base-config', type=str, default=DEFAULT_BASE_CONFIG)
    ap.add_argument('--device', type=str, default=None, help='override device in base config')
    ap.add_argument('--repeats', type=int, default=None, help='number of seeds per activation k')
    ap.add_argument('--dry-run', action='store_true', help='only print planned runs without training')
    args = ap.parse_args()

    # If an activation config file exists in repo and none provided, use it
    if args.activation_config is None and os.path.exists(DEFAULT_ACTIVATION_CONFIG):
        args.activation_config = DEFAULT_ACTIVATION_CONFIG

    # If base-config wasn't explicitly provided but activation config exists and is a full config,
    # default base-config to the activation config so single-file runs work.
    if args.base_config == DEFAULT_BASE_CONFIG and args.activation_config == DEFAULT_ACTIVATION_CONFIG:
        # If default base config file doesn't exist but activation config does, use activation as base
        if not os.path.exists(args.base_config) and os.path.exists(args.activation_config):
            args.base_config = args.activation_config

    # Load base config early so CLI overrides can be applied with correct precedence
    base = load_base_cfg(args.base_config)

    # If an activation-specific config is provided, merge its keys into base
    # (but keep ks/ks_list handling separate). This makes `config_activation.yaml`
    # able to supply base experiment parameters even when a default config.yaml
    # is present.
    if args.activation_config and os.path.exists(args.activation_config):
        try:
            act_cfg_full = load_base_cfg(args.activation_config)
            if isinstance(act_cfg_full, dict):
                for kk, vv in act_cfg_full.items():
                    if kk in ('ks', 'ks_list'):
                        continue
                    # overlay activation config values into base; CLI args will still override later
                    base[kk] = vv
        except Exception as e:
            print(f"[WARN] failed to merge activation config into base: {e}")

    # Determine ks: CLI --ks has highest precedence; then --activation-config; then default
    ks = parse_ks_arg(args.ks) or None
    if ks is None and args.activation_config:
        try:
            with open(args.activation_config, 'r', encoding='utf-8') as f:
                act_cfg = yaml.safe_load(f)
            if isinstance(act_cfg, dict):
                # prefer explicit ks_list YAML list
                if 'ks_list' in act_cfg and act_cfg['ks_list'] is not None:
                    ks = [int(x) for x in act_cfg['ks_list']]
                elif 'ks' in act_cfg and act_cfg['ks'] is not None:
                    # ks might be a string like "2,3,4" or "2-8" or a YAML string like "[2,3,4]"
                    raw = act_cfg['ks']
                    if isinstance(raw, list):
                        ks = [int(x) for x in raw]
                    else:
                        s = str(raw).strip()
                        # if it looks like a bracketed list, strip brackets
                        if s.startswith('[') and s.endswith(']'):
                            s = s[1:-1]
                        ks = parse_ks_arg(s)
        except Exception as e:
            print(f"[WARN] failed to read activation config {args.activation_config}: {e}")
    ks = ks or DEFAULT_KS

    # Decide steps/samples/batch: precedence CLI > base config file > defaults
    steps = args.steps if args.steps is not None else int(base.get('steps', DEFAULT_STEPS))
    samples = args.samples if args.samples is not None else int(base.get('samples', DEFAULT_SAMPLES))
    batch = args.batch_size if args.batch_size is not None else int(base.get('batch_size', DEFAULT_BATCH))

    # ensure base has those fields set to the final values used
    base['steps'] = steps
    base['samples'] = samples
    base['batch_size'] = batch
    geometry = str(base.get('geometry', 'sphere')).lower()
    if geometry not in ('sphere', 'torus'):
        raise ValueError(f"Unsupported geometry '{geometry}' in base config")
    ambient_dim = int(base.get('ambient_dim', 3))
    if args.device:
        base['device'] = args.device

    # decide device using same logic as main script (prefer mps if requested)
    req = base.get('device', 'cpu')
    device = torch.device('cpu')
    if str(req).lower().startswith('mps'):
        try:
            if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                device = torch.device(req)
        except Exception:
            device = torch.device('cpu')

    # (args.activation_config may have been set above)

    ts = int(time.time())
    out_dir = os.path.join('results', f'activation_sweep_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'results.csv')
    rows = []
    raw_rows = []

    base_seed = int(base.get('seed', 123))
    repeats = args.repeats if args.repeats is not None else int(base.get('repeats', 1))

    print(f"Planned activation ks: {ks}")
    print(f"Using device: {device}")
    print(f"Steps={steps}, samples={samples}, batch_size={batch}")
    print(f"Geometry={geometry}, ambient_dim={ambient_dim}")
    if args.dry_run:
        print('Dry run; exiting without training')
        # prepare sanitized base (exclude activation_k) and write sweep_config + header
        sanitized_base = dict(base)
        sanitized_base.pop('activation_k', None)
        sweep_cfg = {
            'timestamp': ts,
            'act_k_list': ks,
            'steps': steps,
            'samples': samples,
            'batch_size': batch,
            'device': str(device),
            'base_config': sanitized_base,
        }
        with open(os.path.join(out_dir, 'sweep_config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(sweep_cfg, f, sort_keys=False, allow_unicode=True)
        # write config_with_header (machine-readable part uses sanitized_base)
        orig_cfg_src = args.base_config
        combined_path = os.path.join(out_dir, 'config_with_header.yaml')
        with open(combined_path, 'w', encoding='utf-8') as fout:
            try:
                with open(orig_cfg_src, 'r', encoding='utf-8') as fr:
                    fout.write(fr.read())
                    fout.write('\n# ---- final used config below (machine-readable; activation_k excluded) ----\n')
            except FileNotFoundError:
                fout.write('# original config not found\n')
            yaml.safe_dump(sanitized_base, fout, sort_keys=False, allow_unicode=True)
        print('Wrote sweep_config and config_with_header to', out_dir)
        return

    # prepare CSV headers for raw and aggregated
    fieldnames_raw = ['activation_k', 'repeat', 'seed', 'final_loss', 'value_mse', 'grad_mse', 'laplace_mse']
    fieldnames_agg = ['activation_k',
                      'value_mse_mean', 'value_mse_std',
                      'grad_mse_mean', 'grad_mse_std',
                      'laplace_mse_mean', 'laplace_mse_std',
                      'final_loss_mean', 'final_loss_std']

    for k in ks:
        per_k_dir = os.path.join(out_dir, f'k_{int(k)}')
        os.makedirs(per_k_dir, exist_ok=True)
        per_k_runs = []
        for rep in range(repeats):
            cfg = dict(base)
            cfg['activation_k'] = int(k)
            cfg['out_dir'] = os.path.join(per_k_dir, f'rep_{rep}')
            os.makedirs(cfg['out_dir'], exist_ok=True)

            seed = base_seed + rep + int(k) * 1000
            cfg['seed'] = seed
            set_random_seeds(seed)

            print(f"Running activation_k={k}, repeat={rep+1}/{repeats} (seed={seed}) on device={device} ...")
            final_loss, history, model, meta = run_one(cfg, device)

            # evaluation: sample points and compare model to target (value, gradient, Laplace)
            eval_n = int(cfg.get('eval_samples', base.get('eval_samples', cfg.get('samples', 2048))))
            if geometry == 'sphere':
                x_eval, w_eval = fibonacci_sphere(eval_n, device=device, dtype=torch.float32,
                                                  ambient_dim=ambient_dim)
                aux_uv = None
                torus_R = torus_r = None
            else:
                torus_R = float(cfg.get('torus_R', base.get('torus_R', 1.5)))
                torus_r = float(cfg.get('torus_r', base.get('torus_r', 0.5)))
                x_eval, w_eval, aux_uv = sample_torus(eval_n, device=device, dtype=torch.float32,
                                                      R=torus_R, r=torus_r)
            x_eval = x_eval.detach()
            w_eval = w_eval.detach()

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
                normals_eval = torus_normals(aux_uv, torus_R, torus_r)
                g_pred = g_pred - (g_pred * normals_eval).sum(dim=-1, keepdim=True) * normals_eval
                g_star = g_star - (g_star * normals_eval).sum(dim=-1, keepdim=True) * normals_eval
            elif geometry == 'sphere':
                P_eval = proj_matrix(x_eval)
                g_pred = torch.einsum('...ij,...j->...i', P_eval, g_pred)
                g_star = torch.einsum('...ij,...j->...i', P_eval, g_star)

            if geometry == 'sphere':
                lb_pred = laplace_beltrami(f_theta, x_eval)
                if laplace_star_fn is not None:
                    with torch.no_grad():
                        lb_star = laplace_star_fn(x_eval)
                else:
                    lb_star = laplace_beltrami(f_star, x_eval)
            else:
                if aux_uv is None:
                    raise RuntimeError('Torus geometry requires uv coordinates for Laplace evaluation')
                lb_pred = torus_laplace_beltrami(f_theta, aux_uv, torus_R, torus_r)
                if laplace_star_fn is not None:
                    with torch.no_grad():
                        lb_star = laplace_star_fn(x_eval)
                else:
                    lb_star = torus_laplace_beltrami(f_star, aux_uv, torus_R, torus_r)

            residual_val = y_pred - y_star
            residual_lap = lb_pred - lb_star
            grad_diff = g_pred - g_star

            value_mse = mse_integral(residual_val, w_eval)
            grad_mse_val = grad_mse_metric(grad_diff, w_eval)
            laplace_mse = mse_integral(residual_lap, w_eval)

            per_k_runs.append({'activation_k': int(k), 'repeat': rep, 'seed': seed,
                               'final_loss': final_loss,
                               'value_mse': value_mse, 'grad_mse': grad_mse_val, 'laplace_mse': laplace_mse})

            raw_rows.append(per_k_runs[-1])

        # aggregate stats for this k
        value_vals = [r['value_mse'] for r in per_k_runs if isinstance(r.get('value_mse', None), float) and math.isfinite(r.get('value_mse', float('nan')))]
        grad_vals = [r['grad_mse'] for r in per_k_runs if isinstance(r.get('grad_mse', None), float) and math.isfinite(r.get('grad_mse', float('nan')))]
        lap_vals = [r['laplace_mse'] for r in per_k_runs if isinstance(r.get('laplace_mse', None), float) and math.isfinite(r.get('laplace_mse', float('nan')))]
        loss_vals = [r['final_loss'] for r in per_k_runs if isinstance(r.get('final_loss', None), float) and math.isfinite(r.get('final_loss', float('nan')))]

        v_mean, v_std = _mean_std(value_vals)
        g_mean, g_std = _mean_std(grad_vals)
        l_mean, l_std = _mean_std(lap_vals)
        loss_mean, loss_std = _mean_std(loss_vals)

        rows.append({'activation_k': int(k),
                     'value_mse_mean': v_mean, 'value_mse_std': v_std,
                     'grad_mse_mean': g_mean, 'grad_mse_std': g_std,
                     'laplace_mse_mean': l_mean, 'laplace_mse_std': l_std,
                     'final_loss_mean': loss_mean, 'final_loss_std': loss_std})

        # write raw CSV incrementally
        raw_csv_path = os.path.join(out_dir, 'results_raw.csv')
        with open(raw_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames_raw)
            writer.writeheader()
            for r in raw_rows:
                writer.writerow(r)

        # write aggregated CSV incrementally
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames_agg)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    # plot using aggregated means and stds (log10 with asymmetric errors)
    ks_plot = [r['activation_k'] for r in rows]
    relu_order_plot = [int(k) - 1 for k in ks_plot]

    final_means = [r.get('final_loss_mean', float('nan')) for r in rows]
    final_stds = [r.get('final_loss_std', 0.0) for r in rows]
    final_y, final_err = _log10_and_err(final_means, final_stds)

    plt.figure()
    ln_final, = plt.plot(relu_order_plot, final_y, '-o', label='final loss ($\\log_{10}$)')
    color_final = ln_final.get_color()
    lower_loss = [y - le for y, le in zip(final_y, final_err[0])]
    upper_loss = [y + ue for y, ue in zip(final_y, final_err[1])]
    plt.fill_between(relu_order_plot, lower_loss, upper_loss, color=color_final, alpha=0.22)
    plt.errorbar(relu_order_plot, final_y, yerr=final_err, fmt='-', color=color_final, capsize=3, elinewidth=1.0)
    plt.xlabel('ReLU order k')
    plt.ylabel('$\\log_{10}(\\text{final loss})$')
    plt.title('Training loss vs ReLU order')
    plt.grid(True, which='both', ls='--')
    plt.savefig(os.path.join(out_dir, 'plot.png'))

    # plot aggregated MSEs with error bands
    value_means = [r.get('value_mse_mean', float('nan')) for r in rows]
    value_stds = [r.get('value_mse_std', 0.0) for r in rows]
    grad_means = [r.get('grad_mse_mean', float('nan')) for r in rows]
    grad_stds = [r.get('grad_mse_std', 0.0) for r in rows]
    lap_means = [r.get('laplace_mse_mean', float('nan')) for r in rows]
    lap_stds = [r.get('laplace_mse_std', 0.0) for r in rows]

    val_y, val_err = _log10_and_err(value_means, value_stds)
    grad_y, grad_err = _log10_and_err(grad_means, grad_stds)
    lap_y, lap_err = _log10_and_err(lap_means, lap_stds)

    plt.figure()
    # value
    ln_val, = plt.plot(relu_order_plot, val_y, '-', label='value MSE ($\\log_{10}$)')
    color_val = ln_val.get_color()
    lower = [y - le for y, le in zip(val_y, val_err[0])]
    upper = [y + ue for y, ue in zip(val_y, val_err[1])]
    plt.fill_between(relu_order_plot, lower, upper, color=color_val, alpha=0.22)
    plt.errorbar(relu_order_plot, val_y, yerr=val_err, fmt='-', color=color_val, capsize=3, elinewidth=1.0)

    # grad (label matches width_sweep style)
    ln_grad, = plt.plot(relu_order_plot, grad_y, '-', label='grad MSE ($\\log_{10}$)')
    color_grad = ln_grad.get_color()
    lower = [y - le for y, le in zip(grad_y, grad_err[0])]
    upper = [y + ue for y, ue in zip(grad_y, grad_err[1])]
    plt.fill_between(relu_order_plot, lower, upper, color=color_grad, alpha=0.22)
    plt.errorbar(relu_order_plot, grad_y, yerr=grad_err, fmt='-', color=color_grad, capsize=3, elinewidth=1.0)

    # lap
    ln_lap, = plt.plot(relu_order_plot, lap_y, '-', label='laplace MSE ($\\log_{10}$)')
    color_lap = ln_lap.get_color()
    lower = [y - le for y, le in zip(lap_y, lap_err[0])]
    upper = [y + ue for y, ue in zip(lap_y, lap_err[1])]
    plt.fill_between(relu_order_plot, lower, upper, color=color_lap, alpha=0.22)
    plt.errorbar(relu_order_plot, lap_y, yerr=lap_err, fmt='-', color=color_lap, capsize=3, elinewidth=1.0)

    plt.xlabel('ReLU order k')
    plt.ylabel('$\\log_{10}(\\mathrm{MSE})$')
    plt.title('Value / Grad / Laplace MSE across ReLU order')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.savefig(os.path.join(out_dir, 'plot_errors.png'))
    # save sweep configuration for reproducibility
    # prepare a sanitized base config that excludes activation_k (sweep varies it)
    sanitized_base = dict(base)
    sanitized_base.pop('activation_k', None)
    sweep_cfg = {
        'timestamp': ts,
        'act_k_list': ks,
        'steps': steps,
        'samples': samples,
        'batch_size': batch,
        'device': str(device),
        'base_config': sanitized_base,
    }
    try:
        with open(os.path.join(out_dir, 'sweep_config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(sweep_cfg, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        print(f"[WARN] failed to write sweep_config.yaml: {e}")

    # also write a combined config_with_header.yaml preserving original comments,
    # but exclude activation_k from the machine-readable section so the sweep is clear
    orig_cfg_src = args.base_config
    combined_path = os.path.join(out_dir, 'config_with_header.yaml')
    try:
        with open(combined_path, 'w', encoding='utf-8') as fout:
            try:
                with open(orig_cfg_src, 'r', encoding='utf-8') as fr:
                    fout.write(fr.read())
                    fout.write('\n# ---- final used config below (machine-readable; activation_k excluded) ----\n')
            except FileNotFoundError:
                fout.write('# original config not found\n')
            yaml.safe_dump(sanitized_base, fout, sort_keys=False, allow_unicode=True)
    except Exception:
        pass

    print('Sweep done ->', out_dir)


if __name__ == '__main__':
    main()
