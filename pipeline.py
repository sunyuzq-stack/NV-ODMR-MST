import os, time, json, copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from config import CFG
from odmr_core import (SyntheticODMRDataset, FiniteSyntheticDataset,
                       generate_single_spectrum,
                       load_odmr_stack, validate_stack_pair,
                       stack_to_pixel_spectra, zscore_spectra,
                       reconstruct_spectrum_np, full_physics_pipeline,
                       generate_synthetic_stack, splitting_to_bz,
                       N_LABELS, LABEL_NAMES)
from nn_model import ODMR_CNN, ODMRLoss, build_model, _PARAM_NAMES


# Learning-rate schedule.


def _lr_schedule(opt, ep, total, lr_max, lr_min, warmup_frac):
    we = int(total * warmup_frac)
    if ep < we:
        lr = lr_max * (ep + 1) / (we + 1)
    else:
        p = (ep - we) / max(total - we, 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * p))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def train(cfg=CFG):
    cfg.ensure_dirs()
    tc = cfg.train
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = tc.use_amp and dev.type == "cuda"
    if tc.use_amp and dev.type != "cuda":
        print("[train] AMP requested but device is not cuda; AMP disabled.")


    seed = int(getattr(cfg, "seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print(f"[train] seed={seed}, run_name={getattr(cfg, 'run_name', 'default')}")


    model, _ = build_model(cfg, dev)
    crit = ODMRLoss(cfg).to(dev)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tc.lr_max,
        weight_decay=tc.weight_decay,
        betas=tc.betas
    )
    scaler = GradScaler('cuda', enabled=use_amp)

    tr_ds = SyntheticODMRDataset(cfg, base_seed=seed)
    va_ds = FiniteSyntheticDataset(
        n=tc.val_batches * tc.batch_size,
        cfg=cfg,
        seed=999999,
    )

    nw = int(getattr(tc, "num_workers", 2))
    tr_dl = DataLoader(tr_ds, batch_size=tc.batch_size,
                       num_workers=nw, pin_memory=True, prefetch_factor=4,
                       persistent_workers=(nw > 0))
    va_dl = DataLoader(va_ds, batch_size=tc.batch_size, shuffle=False,
                       num_workers=1, pin_memory=True, prefetch_factor=2)

    best = float("inf")
    ckpt = os.path.join(cfg.checkpoint_dir, cfg.best_model_name)
    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")


    if os.path.exists(log_path):
        os.remove(log_path)

    print(f"[train] {tc.epochs} epochs, {tc.batches_per_epoch} batches/ep, "
          f"bs={tc.batch_size}, device={dev}, AMP={use_amp}, num_workers={nw}")

    for ep in range(tc.epochs):
        model.train()
        lr = _lr_schedule(opt, ep, tc.epochs, tc.lr_max, tc.lr_min,
                          tc.warmup_fraction)
        el = {}
        t0 = time.time()
        it = iter(tr_dl)
        for _ in range(tc.batches_per_epoch):
            sp, lb = next(it)
            sp, lb = sp.to(dev, non_blocking=True), lb.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=use_amp):
                pred = model(sp)
                loss, ld = crit(pred, lb, input_spec=sp)

            if not torch.isfinite(loss):
                print("[train] non-finite loss, skip batch:", ld)
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                tc.grad_clip,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                print("[train] non-finite grad_norm, skip step")
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(opt)
            scaler.update()
            for k, v in ld.items():
                el[k] = el.get(k, 0) + v
        for k in el:
            el[k] /= tc.batches_per_epoch
        dt = time.time() - t0


        model.eval()
        vl = {}
        vi = iter(va_dl)
        with torch.no_grad():
            for _ in range(tc.val_batches):
                sp, lb = next(vi)
                sp, lb = sp.to(dev), lb.to(dev)
                with autocast('cuda', enabled=use_amp):
                    _, ld = crit(model(sp), lb, input_spec=sp)
                for k, v in ld.items():
                    vl[k] = vl.get(k, 0) + v
        for k in vl:
            vl[k] /= tc.val_batches

        vt = vl.get("total", float("inf"))
        imp = vt < best
        if imp:
            best = vt
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "val_loss": vt, "cfg_n_freq": cfg.n_freq,
                        "n_raw_outputs": cfg.model.n_raw_outputs,


                        "arch": {
                            "conv_layers": [list(t) for t in cfg.model.conv_layers],
                            "fc_hidden": list(cfg.model.fc_hidden),
                            "use_batchnorm": bool(getattr(cfg.model, "use_batchnorm", False)),
                            "dropout": float(getattr(cfg.model, "dropout", 0.0)),
                        }}, ckpt)
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": ep, "lr": lr, "dt": dt,
                                "train": el, "val": vl}) + "\n")
        if ep % 10 == 0 or imp:
            m = " *" if imp else ""
            print(f"  [{ep:3d}/{tc.epochs}] tr_total={el.get('total',0):.5f} "
                  f"va_total={vt:.5f} va_nll={vl.get('nll',0):.5f} "
                  f"lr={lr:.2e} {dt:.1f}s{m}")

    print(f"[train] Best val={best:.6f}  ckpt={ckpt}")


    try:
        from visualize import plot_training_curves
        plot_training_curves(log_path, cfg.output_dir)
        print(f"[train] Training curves saved under {cfg.output_dir}/")
    except Exception as e:
        print(f"[train] WARNING: failed to plot training curves: {e}")

    return ckpt


@torch.no_grad()
def predict_spectra(model, sp_normed, device, bs=512, return_sigma=False):
    model.eval()
    p_dev = next(model.parameters()).device
    if str(device).startswith("cuda") and p_dev.type != "cuda":
        raise RuntimeError(f"Requested device={device} but model on {p_dev}. "
                           f"检查 torch.cuda.is_available() 和 PyTorch CUDA 版本")
    N = sp_normed.shape[0]
    means = []
    sigmas = []
    for i in range(0, N, bs):
        b = torch.from_numpy(sp_normed[i:min(i+bs, N)]).unsqueeze(1).to(
            device, non_blocking=True)
        pred = model(b)
        means.append(pred.to_tensor().cpu().numpy())
        if return_sigma:
            sigmas.append(pred.get_sigma().cpu().numpy())
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    mean_arr = np.concatenate(means, axis=0)
    if return_sigma:
        sigma_arr = np.concatenate(sigmas, axis=0)
        return mean_arr, sigma_arr
    return mean_arr


def load_checkpoint(path, cfg=CFG, device=None):
    if device is None:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    cn = ck.get("cfg_n_freq")
    if cn is not None and cn != cfg.n_freq:
        raise ValueError(f"Checkpoint n_freq={cn} != config {cfg.n_freq}")
    n_raw = ck.get("n_raw_outputs")
    if n_raw is not None and n_raw != cfg.model.n_raw_outputs:
        raise ValueError(
            f"Checkpoint output dim={n_raw} != config {cfg.model.n_raw_outputs}. "
            f"This checkpoint was likely trained with a different output schema "
            f"(e.g. without probabilistic outputs); please retrain.")


    arch = ck.get("arch")
    build_cfg = cfg
    if arch is not None:
        cur = {"conv_layers": [list(t) for t in cfg.model.conv_layers],
               "fc_hidden": list(cfg.model.fc_hidden),
               "use_batchnorm": bool(getattr(cfg.model, "use_batchnorm", False))}
        ckp = {"conv_layers": [list(t) for t in arch.get("conv_layers", [])],
               "fc_hidden": list(arch.get("fc_hidden", [])),
               "use_batchnorm": bool(arch.get("use_batchnorm", False))}
        if ckp != cur:
            print(f"[pipeline] WARNING: checkpoint arch != current config arch; "
                  f"rebuilding from checkpoint arch.\n"
                  f"  checkpoint: {ckp}\n  config   : {cur}")
        build_cfg = copy.deepcopy(cfg)
        build_cfg.model.conv_layers = [tuple(t) for t in arch["conv_layers"]]
        build_cfg.model.fc_hidden = list(arch["fc_hidden"])
        if "use_batchnorm" in arch:
            build_cfg.model.use_batchnorm = bool(arch["use_batchnorm"])
        if "dropout" in arch:
            build_cfg.model.dropout = float(arch["dropout"])

    model = ODMR_CNN(build_cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"[pipeline] Loaded epoch {ck.get('epoch','?')} "
          f"val_loss={ck.get('val_loss',0):.6f}")
    return model, device


def compute_residual_map(pred_2d, normed_spectra, freq_ghz, H, W, chunk=512):
    N = H * W
    p = np.asarray(pred_2d, dtype=np.float64).reshape(N, 10)
    obs = np.asarray(normed_spectra, dtype=np.float64).reshape(N, -1)
    f = np.asarray(freq_ghz, dtype=np.float64)
    out = np.empty(N, dtype=np.float32)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        q = p[s:e]
        c = [q[:, i][:, None] for i in range(10)]
        recon = reconstruct_spectrum_np(
            f, c[0], c[1], c[2], c[3], c[4], c[5],
            c[6], c[7], c[8], c[9])
        mu = recon.mean(axis=1, keepdims=True)
        sd = np.maximum(recon.std(axis=1, keepdims=True), 1e-12)
        rz = (recon - mu) / sd
        out[s:e] = np.sqrt(np.mean((rz - obs[s:e]) ** 2, axis=1))

    return out.reshape(H, W)


def compute_confidence_map(residual_map, invalid_mask=None, cfg=CFG):
    cc = cfg.confidence
    res = residual_map.copy().astype(np.float64)

    if invalid_mask is not None:
        valid_res = res[~invalid_mask]
    else:
        valid_res = res.ravel()

    if len(valid_res) == 0:
        return np.zeros_like(res, dtype=np.float32)

    q_lo = float(np.percentile(valid_res, cc.quantile_lo * 100))
    q_hi = float(np.percentile(valid_res, cc.quantile_hi * 100))

    if cc.method in ("quantile", "quantile_exp"):
        dq = max(q_hi - q_lo, 1e-12)
        res_norm = np.clip((res - q_lo) / dq, 0.0, 1.0)
    else:
        res_max = max(res.max(), 1e-12)
        res_norm = res / res_max

    if cc.method == "quantile":
        conf = 1.0 - res_norm
    elif cc.method == "exp":
        conf = np.exp(-cc.alpha * res_norm)
    else:  # quantile_exp
        conf = np.exp(-cc.alpha * res_norm)

    if invalid_mask is not None:
        conf[invalid_mask] = 0.0
    return conf.astype(np.float32)


def infer_stack(stack_path, ckpt_path, zerofield_path=None, delta0_mhz=0.0,
                cfg=CFG, save_dir=None,
                reduce_repeats="mean", expected_n_freq=None,
                strict_stack_shape=True):
    if save_dir is None:
        save_dir = cfg.output_dir
    os.makedirs(save_dir, exist_ok=True)

    model, dev = load_checkpoint(ckpt_path, cfg)
    stk = load_odmr_stack(stack_path,
                          expected_n_freq=expected_n_freq,
                          reduce_repeats=reduce_repeats,
                          strict=strict_stack_shape)
    Nf, H, W = stk.shape


    pred_zf_2d = None
    if zerofield_path is not None:
        zf = load_odmr_stack(zerofield_path,
                             expected_n_freq=expected_n_freq,
                             reduce_repeats=reduce_repeats,
                             strict=strict_stack_shape)
        validate_stack_pair(stk, zf, cfg.n_freq)
        sp_zf, _, _ = stack_to_pixel_spectra(zf)
        nm_zf = zscore_spectra(sp_zf)
        pred_zf_mean, _ = predict_spectra(
            model, nm_zf, dev, cfg.infer_batch_size, return_sigma=True)
        pred_zf_2d = pred_zf_mean.reshape(H, W, -1)


    sp, _, _ = stack_to_pixel_spectra(stk)
    nm = zscore_spectra(sp)
    pred_mean, pred_sigma = predict_spectra(
        model, nm, dev, cfg.infer_batch_size, return_sigma=True)
    pred_2d = pred_mean.reshape(H, W, -1)
    sigma_2d = pred_sigma.reshape(H, W, -1)


    result = full_physics_pipeline(pred_2d, pred_zf_2d, delta0_mhz,
                                   cfg.gamma_e_mhz_per_gauss, cfg)
    result.update({
        "H": H, "W": W,
        "pred_raw": pred_2d,
        "sigma_raw": sigma_2d,

        "center_sigma_ghz": sigma_2d[..., 0],
        "split_sigma_mhz": sigma_2d[..., 1],
        "w1_sigma_mhz": sigma_2d[..., 2],
    })


    res = compute_residual_map(pred_2d, nm, cfg.freq_axis_ghz, H, W)
    result["residual_map"] = res

    inv = ~result["valid_mask"]
    result["confidence_map"] = compute_confidence_map(res, invalid_mask=inv, cfg=cfg)


    sp_sigma = sigma_2d[..., 1]
    sp_sigma_clip = np.clip(sp_sigma, 1e-6, None)
    sigma_conf = 1.0 / (1.0 + sp_sigma_clip / np.median(sp_sigma_clip[~inv])) \
                 if (~inv).any() else np.zeros_like(sp_sigma)
    result["model_confidence_map"] = sigma_conf.astype(np.float32)


    save_keys = ["splitting_mhz", "bz_gauss", "pred_raw", "sigma_raw",
                 "valid_mask", "center_ghz", "w1_mhz", "w2_mhz",
                 "bl_offset", "bl_slope", "residual_map", "confidence_map",
                 "model_confidence_map", "center_sigma_ghz", "split_sigma_mhz",
                 "w1_sigma_mhz"]
    for k in save_keys:
        if k in result:
            np.save(os.path.join(save_dir, f"{k}.npy"), result[k])

    v = result["valid_mask"]
    print(f"[infer] Invalid: {(~v).sum()}/{H*W} ({(~v).mean()*100:.1f}%)")
    print(f"[infer] Mean residual: {res.mean():.4f}")
    print(f"[infer] Center sigma median (MHz): "
          f"{np.median(sigma_2d[..., 0]) * 1e3:.3f}")
    print(f"[infer] Split sigma median (MHz): "
          f"{np.median(sigma_2d[..., 1]):.3f}")
    print("[infer] Primary maps: center_ghz, splitting_mhz, w1_mhz, w2_mhz, residual_map. "
          "Auxiliary diagnostics: c1/c2/eta/baseline/confidence/sigma_*.")
    return result


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _snr_binned_rmse(pred, true, snrs,
                     bins=[(3,7),(7,12),(12,20),(20,35),(35,60),(60,100),(100,500)]):
    r = {}
    for lo, hi in bins:
        m = (snrs >= lo) & (snrs < hi)
        if m.sum():
            r[(lo, hi)] = {"rmse": _rmse(pred[m], true[m]), "n": int(m.sum())}
    return r


def _gen_test(cfg, n, seed):
    rng = np.random.default_rng(seed)
    sp, lb, sn = [], [], []
    for _ in range(n):
        s, l, snr, _ = generate_single_spectrum(rng, cfg)
        sp.append(s); lb.append(l); sn.append(snr)
    return np.stack(sp), np.stack(lb), np.array(sn)


def _core_eval_metrics(pred, true):
    pred_width_mean = 0.5 * (pred[:, 2] + pred[:, 3])
    true_width_mean = 0.5 * (true[:, 2] + true[:, 3])
    return {
        "rmse_center_mhz": _rmse(pred[:, 0], true[:, 0]) * 1e3,
        "rmse_split_mhz": _rmse(pred[:, 1], true[:, 1]),
        "rmse_w1_mhz": _rmse(pred[:, 2], true[:, 2]),
        "rmse_w2_mhz": _rmse(pred[:, 3], true[:, 3]),
        "rmse_width_mean_mhz": _rmse(pred_width_mean, true_width_mean),
        "errors_center_mhz": (pred[:, 0] - true[:, 0]) * 1e3,
        "errors_split_mhz": pred[:, 1] - true[:, 1],
        "errors_width_mean_mhz": pred_width_mean - true_width_mean,
    }


def _calibration_stats(pred_mean, pred_sigma, true):
    z = (pred_mean - true) / np.clip(pred_sigma, 1e-12, None)
    out = {}
    for i, n in enumerate(_PARAM_NAMES):
        out[f"std_resid_mean_{n}"] = float(z[:, i].mean())
        out[f"std_resid_std_{n}"] = float(z[:, i].std())
        out[f"sigma_median_{n}"] = float(np.median(pred_sigma[:, i]))
    return out, z


def evaluate_id(model, dev, cfg=CFG, n=5000, seed=77777):
    print(f"[eval-ID] {n} spectra...")
    sp, lb, sn = _gen_test(cfg, n, seed)
    pr_mean, pr_sigma = predict_spectra(
        model, sp, dev, cfg.infer_batch_size, return_sigma=True)
    m = _core_eval_metrics(pr_mean, lb)
    cal_stats, z_resid = _calibration_stats(pr_mean, pr_sigma, lb)
    m.update(cal_stats)
    print(f"[eval-ID] center RMSE={m['rmse_center_mhz']:.4f} MHz, "
          f"split RMSE={m['rmse_split_mhz']:.4f} MHz, "
          f"width(mean) RMSE={m['rmse_width_mean_mhz']:.4f} MHz")
    print(f"[eval-ID] σ calibration (target=1.0): "
          f"center={m['std_resid_std_center']:.3f}, "
          f"split={m['std_resid_std_split']:.3f}")
    return {"pred": pr_mean, "sigma": pr_sigma, "true": lb, "snrs": sn,
            "std_resid": z_resid, **m,
            "rmse_per_snr": _snr_binned_rmse(pr_mean[:, 0] * 1e3, lb[:, 0] * 1e3, sn),
            "rmse_per_snr_split_mhz": _snr_binned_rmse(pr_mean[:, 1], lb[:, 1], sn)}


def evaluate_ood(model, dev, cfg=CFG, n=5000, seed=88888):
    oc = copy.deepcopy(cfg)
    oc.noise.photon_count_range = (50_000, 500_000)
    oc.noise.readout_noise_std = 0.005
    oc.noise.multiplicative_noise_std = 0.01
    oc.noise.bad_pixel_prob = 0.01
    oc.baseline.slope_range = (-0.05, 0.05)
    oc.baseline.quad_range = (-0.03, 0.03)
    oc.baseline.slow_bg_amplitude = 0.015
    oc.freq_perturb.shift_mhz_std = 1.0
    oc.lineshape.width_asym_max = 0.4
    oc.lineshape.contrast_asym_max = 0.4
    print(f"[eval-OOD] {n} spectra (harsh conditions)...")
    sp, lb, sn = _gen_test(oc, n, seed)
    pr_mean, pr_sigma = predict_spectra(
        model, sp, dev, cfg.infer_batch_size, return_sigma=True)
    m = _core_eval_metrics(pr_mean, lb)
    cal_stats, z_resid = _calibration_stats(pr_mean, pr_sigma, lb)
    m.update(cal_stats)
    print(f"[eval-OOD] center RMSE={m['rmse_center_mhz']:.4f} MHz, "
          f"split RMSE={m['rmse_split_mhz']:.4f} MHz, "
          f"width(mean) RMSE={m['rmse_width_mean_mhz']:.4f} MHz")
    return {"pred": pr_mean, "sigma": pr_sigma, "true": lb, "snrs": sn,
            "std_resid": z_resid, **m,
            "rmse_per_snr": _snr_binned_rmse(pr_mean[:, 0] * 1e3, lb[:, 0] * 1e3, sn),
            "rmse_per_snr_split_mhz": _snr_binned_rmse(pr_mean[:, 1], lb[:, 1], sn)}


def save_eval(results, label, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    s = {
        "metric_interpretation": {
            "primary": [
                "center_ghz / rmse_center_mhz",
                "split_mhz / rmse_split_mhz",
                "w1_mhz / rmse_w1_mhz",
                "w2_mhz / rmse_w2_mhz",
                "width_mean / rmse_width_mean_mhz",
                "residual_map",
            ],
            "auxiliary": [
                "c1", "c2", "eta", "bl_offset", "bl_slope",
                "bl_quad", "confidence_map", "sigma_*", "std_resid_*",
            ],
        }
    }
    for k in [
        "rmse_center_mhz", "rmse_split_mhz", "rmse_w1_mhz",
        "rmse_w2_mhz", "rmse_width_mean_mhz",
    ]:
        if k in results:
            s[k] = results[k]

    for k in list(results.keys()):
        if k.startswith("std_resid_") or k.startswith("sigma_median_"):
            s[k] = results[k]

    if "rmse_per_snr" in results:
        s["rmse_per_snr"] = {f"{a}-{b}": v
                             for (a, b), v in results["rmse_per_snr"].items()}
    if "rmse_per_snr_split_mhz" in results:
        s["rmse_per_snr_split_mhz"] = {f"{a}-{b}": v
                                       for (a, b), v in results["rmse_per_snr_split_mhz"].items()}
    with open(os.path.join(save_dir, f"{label}_summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    np.savez(os.path.join(save_dir, f"{label}_arrays.npz"),
             pred=results["pred"], sigma=results.get("sigma", np.array([])),
             true=results["true"], snrs=results.get("snrs", np.array([])),
             std_resid=results.get("std_resid", np.array([])))
    print(f"[eval] Saved {label} to {save_dir}/")


def evaluate_synthetic_stack(model, dev, cfg=CFG, H=64, W=64, seed=2026, save_dir=None):
    print(f"[synth-eval] Generating {H}x{W} synthetic stack...")
    gt = generate_synthetic_stack(cfg, H=H, W=W, seed=seed)
    stk = gt["stack"]
    sp, _, _ = stack_to_pixel_spectra(stk)
    nm = zscore_spectra(sp)

    print(f"[synth-eval] Running CNN inference (with sigma)...")
    pred_mean, pred_sigma = predict_spectra(
        model, nm, dev, cfg.infer_batch_size, return_sigma=True)
    pred_2d = pred_mean.reshape(H, W, -1)
    sigma_2d = pred_sigma.reshape(H, W, -1)


    snr_map = gt["snr_per_pixel"]
    photons_map = gt["photons_per_pixel"]


    metrics = {}
    pairs = [
        ("center_ghz", 0, "center_ghz_gt", 1e3, "center_rmse_mhz"),
        ("split_mhz", 1, "split_mhz_gt", 1.0, "split_rmse_mhz"),
        ("w1_mhz", 2, "w1_mhz_gt", 1.0, "w1_rmse_mhz"),
        ("w2_mhz", 3, "w2_mhz_gt", 1.0, "w2_rmse_mhz"),
        ("c1", 4, "c1_gt", 1.0, "c1_rmse"),
        ("c2", 5, "c2_gt", 1.0, "c2_rmse"),
        ("eta", 6, "eta_gt", 1.0, "eta_rmse"),
        ("bl_offset", 7, "bl_offset_gt", 1.0, "bl_offset_rmse"),
    ]
    for name, idx, gt_key, scale, metric_key in pairs:
        if gt_key in gt:
            p = pred_2d[..., idx] * scale
            g = gt[gt_key] * scale
            metrics[metric_key] = _rmse(p, g)

    pred_split = pred_2d[..., 1]
    bz_pred = pred_split / (2.0 * cfg.gamma_e_mhz_per_gauss)
    bz_gt = gt["bz_gauss_gt"]
    metrics["bz_rmse_gauss"] = _rmse(bz_pred, bz_gt)

    res = compute_residual_map(pred_2d, nm, cfg.freq_axis_ghz, H, W)
    metrics["residual_mean"] = float(res.mean())
    metrics["residual_std"] = float(res.std())
    metrics["residual_p95"] = float(np.percentile(res, 95))


    bin_metrics = {}
    for lo, hi, name in cfg.eval.snr_bins:
        mask = (snr_map >= lo) & (snr_map < hi)
        n_pix = int(mask.sum())
        if n_pix < 5:
            continue
        bm = {"n_pixels": n_pix,
              "snr_range": [float(lo), float(hi)],
              "snr_actual_mean": float(snr_map[mask].mean()),
              "photons_mean": float(photons_map[mask].mean())}

        for pname, idx, gt_key, scale, _ in pairs:
            if gt_key in gt:
                p = pred_2d[..., idx][mask] * scale
                g = gt[gt_key][mask] * scale
                bm[f"{pname}_rmse"] = float(_rmse(p, g))
        # Bz
        bm["bz_rmse_gauss"] = float(_rmse(bz_pred[mask], bz_gt[mask]))

        bm["sigma_center_median_mhz"] = float(np.median(sigma_2d[..., 0][mask]) * 1e3)
        bm["sigma_split_median_mhz"] = float(np.median(sigma_2d[..., 1][mask]))
        bin_metrics[name] = bm

    metrics["by_snr"] = bin_metrics
    metrics["snr_map_min"] = float(snr_map.min())
    metrics["snr_map_max"] = float(snr_map.max())
    metrics["snr_map_mean"] = float(snr_map.mean())
    metrics["snr_map_median"] = float(np.median(snr_map))


    print(f"[synth-eval] Global metrics:")
    for k in ["center_rmse_mhz", "split_rmse_mhz", "w1_rmse_mhz", "w2_rmse_mhz", "bz_rmse_gauss", "residual_mean", "residual_p95"]:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.6f}")

    print(f"[synth-eval] By-SNR breakdown:")
    print(f"  SNR map: min={metrics['snr_map_min']:.2f} "
          f"median={metrics['snr_map_median']:.2f} "
          f"max={metrics['snr_map_max']:.2f}")
    for name, bm in bin_metrics.items():
        print(f"  [{name}] n={bm['n_pixels']:>5d} "
              f"(SNR≈{bm['snr_actual_mean']:.1f}): "
              f"center={bm['center_ghz_rmse']:.3f}MHz | "
              f"split={bm['split_mhz_rmse']:.3f}MHz | "
              f"σ_split_med={bm['sigma_split_median_mhz']:.3f}MHz")

    result = {
        "metrics": metrics,
        "pred_2d": pred_2d,
        "sigma_2d": sigma_2d,
        "gt": gt,
        "residual_map": res,
        "bz_pred": bz_pred,
        "snr_map": snr_map,
        "H": H, "W": W,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "synth_eval_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        np.save(os.path.join(save_dir, "synth_pred_2d.npy"), pred_2d)
        np.save(os.path.join(save_dir, "synth_sigma_2d.npy"), sigma_2d)
        np.save(os.path.join(save_dir, "synth_residual_map.npy"), res)
        np.save(os.path.join(save_dir, "synth_snr_map.npy"), snr_map)
        print(f"[synth-eval] Saved to {save_dir}/")

    return result
