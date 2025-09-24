
import argparse, yaml, os, time, math
import torch
from train.trainer import run_training
from test.evaluator import evaluate_model
from utils.viz import save_learning_curve

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--s", type=int, dest="s_order", default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    return ap.parse_args()

def merge_cfg(cfg, args):
    for k,v in vars(args).items():
        if v is not None:
            cfg[k]=v
    return cfg

if __name__=="__main__":
    # read config as UTF-8 to avoid platform-dependent decoding issues
    with open("config.yaml","r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args = parse_args()
    cfg = merge_cfg(cfg, args)

    torch.manual_seed(cfg.get("seed",123))
    # Safe device selection: support 'cpu', 'cuda' (e.g. 'cuda' or 'cuda:0')
    # and 'mps' (e.g. 'mps' or 'mps:0'). Only create a device with CUDA/MPS
    # after checking availability to avoid lazy-init errors. If the
    # requested backend is not available, fall back to CPU and print info.
    req_dev = str(cfg.get("device", "cpu"))
    req_low = req_dev.lower()

    def _mps_available():
        try:
            return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        except Exception:
            return False

    if req_low == "cpu":
        device = torch.device("cpu")
    elif req_low.startswith("cuda") or req_low == "gpu":
        # treat 'gpu' as request for CUDA
        try:
            if torch.cuda.is_available():
                device = torch.device(req_dev)
            else:
                print(f"[INFO] CUDA requested ({req_dev}) but not available; falling back to CPU.")
                device = torch.device("cpu")
        except Exception:
            # In case torch.cuda.* access raises, fallback safely
            print(f"[INFO] Error checking CUDA availability for '{req_dev}'; falling back to CPU.")
            device = torch.device("cpu")
    elif req_low.startswith("mps"):
        if _mps_available():
            device = torch.device(req_dev)
        else:
            print(f"[INFO] MPS requested ({req_dev}) but not available; falling back to CPU.")
            device = torch.device("cpu")
    else:
        # For any other string, attempt to create a torch.device; if that
        # triggers an error it will surface normally.
        device = torch.device(req_dev)
    # Diagnostic prints
    try:
        print(f"Requested device (from config/args): {req_dev}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}")
    except Exception:
        pass
    os.makedirs("results", exist_ok=True)

    # construct output directory name with descriptive fields and timestamp
    ts = int(time.time())
    task_name = cfg.get('task', 'task')
    target_name = cfg.get('target', 'target')
    s_order = cfg.get('s_order', cfg.get('s', 's'))
    activation_k = cfg.get('activation_k', cfg.get('k', 'k'))
    width = cfg.get('width', 'w')
    depth = cfg.get('depth', 'd')
    geometry = str(cfg.get('geometry', 'geom'))
    out_dir_name = f"{geometry}_{task_name}_{target_name}_s{s_order}_k{activation_k}_w{width}_d{depth}_{ts}"
    out_dir = os.path.join('results', out_dir_name)
    cfg['out_dir'] = out_dir
    history, model, meta = run_training(cfg, device)
    out_dir = meta.get("out_dir", "results")
    # Save the merged runtime config and a copy of the original config
    # into the output directory for reproducibility.
    try:
        os.makedirs(out_dir, exist_ok=True)
        used_cfg_path = os.path.join(out_dir, "config_used.yaml")
        with open(used_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        orig_cfg_src = "config.yaml"
        orig_cfg_dst = os.path.join(out_dir, "config_original.yaml")
        try:
            # copy original config if present
            with open(orig_cfg_src, "r", encoding="utf-8") as fr, open(orig_cfg_dst, "w", encoding="utf-8") as fw:
                fw.write(fr.read())
        except FileNotFoundError:
            # skip if original not found
            pass
        # keep only config_used.yaml and config_original.yaml for standard runs
    except Exception as e:
        print(f"[WARN] Failed to write config files to output dir {out_dir}: {e}")
    save_learning_curve(history, os.path.join(out_dir, "curve.png"))
    evaluate_model(cfg, device, model, meta)
    print("[DONE] outputs ->", out_dir)
