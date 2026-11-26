import os, copy, time
import numpy as np
import torch
from config import CFG, SPARSE_ARCH_BY_K, sparse_arch_fallback
from odmr_core import (load_odmr_stack, stack_to_pixel_spectra, zscore_spectra,splitting_to_bz, params_to_maps, generate_single_spectrum)
from nn_model import ODMR_CNN
from pipeline import train, load_checkpoint, predict_spectra
# Original frequency grid.
FULL_N     = 500
FULL_START = 2.8200  # GHz.
FULL_STEP  = 0.0002  # GHz per frequency step (0.2 MHz).


def sparse_indices(k, full_n=FULL_N):
    return np.round(np.linspace(0, full_n - 1, k)).astype(np.int64)


class SparseFreqAxis:
    def __init__(self, k, full_n=FULL_N, start_ghz=FULL_START, step_ghz=FULL_STEP):
        self.k = int(k); self.full_n = int(full_n)
        self.start_ghz = float(start_ghz); self.step_ghz = float(step_ghz)
        self.n_points = int(k)
    @property
    def sample_idx(self):
        return sparse_indices(self.k, self.full_n)
    @property
    def _full_axis(self):
        return np.linspace(self.start_ghz,self.start_ghz + (self.full_n - 1) * self.step_ghz,self.full_n)
    @property
    def axis_ghz(self):  return self._full_axis[self.sample_idx]
    @property
    def axis_mhz(self):  return self.axis_ghz * 1e3
    @property
    def end_ghz(self):   return float(self.axis_ghz[-1])
    @property
    def span_mhz(self):
        a = self.axis_ghz; return float((a[-1] - a[0]) * 1e3)
    def validate(self):  assert 1 < self.k <= self.full_n


def get_arch_for_k(k):
    if k in SPARSE_ARCH_BY_K:
        conv, fc = SPARSE_ARCH_BY_K[k]
    else:
        conv, fc = sparse_arch_fallback(k)
        print(f"[sparse_fast] WARNING: K={k} 不在 config.SPARSE_ARCH_BY_K，"
              f"用区间规则兜底；建议在该表中加一行 {k}: (...)。")
    return [tuple(t) for t in conv], list(fc)


def make_sparse_cfg(k, base=CFG, bigger=True, infer_bs=4096, seed=None):
    cfg = copy.deepcopy(base)
    cfg.freq = SparseFreqAxis(k)
    if bigger:
        cfg.model.conv_layers, cfg.model.fc_hidden = get_arch_for_k(k)
    cfg.infer_batch_size = infer_bs
    if seed is not None: cfg.seed = seed
    cfg.run_name      = f"fast_K{k}"
    cfg.output_dir    = f"outputs_fast/K{k}"
    cfg.checkpoint_dir= f"checkpoints_fast/K{k}"
    return cfg


def make_variant_cfg(name, base=CFG, calibration=None,
                     target_samples=None, **train_overrides):
    import math
    from config import SPARSE_VARIANTS
    if name not in SPARSE_VARIANTS:
        raise KeyError(f"未知变体 {name}；可选：{list(SPARSE_VARIANTS)}")
    v = SPARSE_VARIANTS[name]
    k = int(v["k"])

    cfg = make_sparse_cfg(k, base=base)
    cfg.model.conv_layers = [tuple(t) for t in v["conv"]]
    cfg.model.fc_hidden   = list(v["fc"])
    for key, val in {**v.get("train", {}), **train_overrides}.items():
        setattr(cfg.train, key, val)
    ts = target_samples if target_samples is not None else v.get("target_samples")
    if ts:
        cfg.train.batches_per_epoch = max(
            1, math.ceil(ts / (cfg.train.epochs * cfg.train.batch_size)))
    if calibration:
        cfg.apply_calibration(calibration)
        cfg.freq = SparseFreqAxis(k)

    if "realism" in v:
        cfg.apply_realism_level(v["realism"])

    for path, val in v.get("cfg_overrides", {}).items():
        obj, parts = cfg, path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
        print(f"[variant] override {path} = {val}")

    if "seed" in v:
        cfg.seed = int(v["seed"])

    cfg.run_name       = f"fast_{name}"
    cfg.output_dir     = f"outputs_fast/{name}"
    cfg.checkpoint_dir = f"checkpoints_fast/{name}"
    return cfg


def train_variant(name, calibration=None, target_samples=None, **train_overrides):
    cfg = make_variant_cfg(name, calibration=calibration,target_samples=target_samples, **train_overrides)
    n = cfg.train.epochs * cfg.train.batches_per_epoch * cfg.train.batch_size
    print(f"[train_variant] {name}: K={cfg.n_freq}")
    print(f"    conv={cfg.model.conv_layers}")
    print(f"    fc  ={cfg.model.fc_hidden}")
    print(f"    batch={cfg.train.batch_size}  lr_max={cfg.train.lr_max:.2e}  "
          f"epochs={cfg.train.epochs}x{cfg.train.batches_per_epoch}  "
          f"样本≈{n:.2e}  AMP={cfg.train.use_amp}")
    print(f"    → {cfg.checkpoint_dir}/ ")
    return train(cfg)


def downsample_stack_file(stack_path, k, out_path, full_n=FULL_N, reduce_repeats="mean"):
    stk = load_odmr_stack(stack_path, expected_n_freq=full_n,
                          reduce_repeats=reduce_repeats, strict=True)
    stk_k = stk[sparse_indices(k, full_n)]
    np.save(out_path, stk_k.astype(np.float32))
    print(f"[downsample] {stk.shape} -> K={k} {stk_k.shape}  saved {out_path}")
    return stk_k


def train_fast(k, calibration=None, seed=None, epochs=None):
    cfg = make_sparse_cfg(k, seed=seed)
    if calibration:
        cfg.apply_calibration(calibration)
        cfg.freq = SparseFreqAxis(k)
    if epochs is not None: cfg.train.epochs = epochs
    print(f"[train_fast] K={k}  axis: {cfg.freq.axis_ghz[0]:.4f}..{cfg.freq.axis_ghz[-1]:.4f} GHz, "
          f"Δf≈{np.diff(cfg.freq.axis_mhz).mean():.3f} MHz, n={cfg.n_freq}")
    return train(cfg)


def fast_infer_stack(stack_path, ckpt_path, k, save_dir,
                     zerofield_path=None, delta0_mhz=0.0,
                     full_n=FULL_N, reduce_repeats="mean",
                     already_sparse=False, infer_bs=4096):
    os.makedirs(save_dir, exist_ok=True)
    cfg = make_sparse_cfg(k, infer_bs=infer_bs)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model, dev = load_checkpoint(ckpt_path, cfg, device=device)

    def _prep(path):
        exp = k if already_sparse else full_n
        stk = load_odmr_stack(path, expected_n_freq=exp, reduce_repeats=reduce_repeats, strict=True)
        if not already_sparse:
            stk = stk[sparse_indices(k, full_n)]
        Nf, H, W = stk.shape
        sp, _, _ = stack_to_pixel_spectra(stk)
        return zscore_spectra(sp).astype(np.float32), H, W

    t1 = time.time(); nm, H, W = _prep(stack_path); t2 = time.time()
    mean, sigma = predict_spectra(model, nm, dev, bs=cfg.infer_batch_size, return_sigma=True)
    t3 = time.time()
    pred = mean.reshape(H, W, 10)
    split = params_to_maps(pred, cfg)["split_mhz"]
    sig_split = sigma.reshape(H, W, 10)[..., 1]
    if zerofield_path:
        nz, Hz, Wz = _prep(zerofield_path)
        mz, _ = predict_spectra(model, nz, dev, bs=cfg.infer_batch_size, return_sigma=True)
        d0 = params_to_maps(mz.reshape(Hz, Wz, 10), cfg)["split_mhz"]
    else:
        d0 = delta0_mhz
    bz, valid = splitting_to_bz(split, d0, cfg.gamma_e_mhz_per_gauss)
    t4 = time.time()

    for n, a in [("split_mhz", split), ("split_sigma_mhz", sig_split),
                 ("bz_gauss", bz), ("valid_mask", valid)]:
        np.save(os.path.join(save_dir, n + ".npy"), a)
    print(f"[fast_infer] K={k} pixels={H*W}  prep={t2-t1:.3f}s "
          f"CNN={t3-t2:.3f}s phys={t4-t3:.3f}s")
    return {"bz_gauss": bz, "split_mhz": split,
            "split_sigma_mhz": sig_split, "valid_mask": valid}


if __name__ == "__main__":
    import io, contextlib
    rng = np.random.default_rng(0)
    print(f"{'K':>5} | {'Δf(MHz)':>8} | {'轴范围(GHz)':>20} | {'谱长':>5} | {'卷积长':>6} | {'参数量':>11} | 前向")
    for k in [500, 400, 300, 200, 100, 80]:
        cfg = make_sparse_cfg(k)
        ax = cfg.freq.axis_ghz
        df = float(np.diff(cfg.freq.axis_mhz).mean())
        spec, lab, snr, meta = generate_single_spectrum(rng, cfg)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m = ODMR_CNN(cfg); m.eval()
        convlen = buf.getvalue().split("conv output length:")[1].split(",")[0].strip()
        npar = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            x = torch.from_numpy(spec).float().view(1, 1, -1)
            out = m(x)
        ok = f"OK split={out.split_mhz.item():.1f}MHz σ={out.get_sigma()[0,1].item():.2f}"
        print(f"{k:>5} | {df:>8.3f} | {ax[0]:.4f}..{ax[-1]:.4f} | {spec.shape[0]:>5} | "
              f"{convlen:>6} | {npar:>11,} | {ok}")
