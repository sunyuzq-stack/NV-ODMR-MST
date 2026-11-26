import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
from config import CFG

# Parameter order shared by the model and the forward spectrum.

LABEL_NAMES = [
    "center_ghz", "split_mhz", "w1_mhz", "w2_mhz",
    "c1", "c2", "eta",
    "bl_offset", "bl_slope", "bl_quad",
]
N_LABELS = len(LABEL_NAMES)
_LN2 = 0.6931471805599453
_SQRT_2LN2 = np.sqrt(2.0 * _LN2)


# Forward spectrum in NumPy.


def reconstruct_spectrum_np(freq_ghz, center_ghz, split_mhz,
                            w1_mhz, w2_mhz, c1, c2, eta,
                            bl_offset=1.0, bl_slope=0.0, bl_quad=0.0):
    freq_mhz = freq_ghz * 1e3
    center_mhz = center_ghz * 1e3
    x1 = center_mhz - split_mhz / 2.0
    x2 = center_mhz + split_mhz / 2.0

    t = (freq_ghz - freq_ghz[0]) / (freq_ghz[-1] - freq_ghz[0])
    bl = bl_offset + bl_slope * (t - 0.5) + bl_quad * (t - 0.5) ** 2

    sig1 = w1_mhz / _SQRT_2LN2
    sig2 = w2_mhz / _SQRT_2LN2
    L1 = 1.0 / (1.0 + ((freq_mhz - x1) / w1_mhz) ** 2)
    G1 = np.exp(-0.5 * ((freq_mhz - x1) / sig1) ** 2)
    L2 = 1.0 / (1.0 + ((freq_mhz - x2) / w2_mhz) ** 2)
    G2 = np.exp(-0.5 * ((freq_mhz - x2) / sig2) ** 2)

    dip1 = c1 * (eta * L1 + (1.0 - eta) * G1)
    dip2 = c2 * (eta * L2 + (1.0 - eta) * G2)
    return bl * (1.0 - dip1 - dip2)


# Differentiable forward spectrum in PyTorch.


def reconstruct_spectrum_torch(freq_mhz, center_ghz, split_mhz,
                               w1_mhz, w2_mhz, c1, c2, eta,
                               bl_offset, bl_slope, bl_quad):
    f = freq_mhz.unsqueeze(0)
    Nf = freq_mhz.shape[0]
    cm = center_ghz.unsqueeze(1) * 1e3
    hs = split_mhz.unsqueeze(1) / 2.0
    x1 = cm - hs
    x2 = cm + hs

    t = torch.linspace(0, 1, Nf, device=freq_mhz.device).unsqueeze(0)
    bl = (bl_offset.unsqueeze(1)
          + bl_slope.unsqueeze(1) * (t - 0.5)
          + bl_quad.unsqueeze(1) * (t - 0.5) ** 2)

    w1 = w1_mhz.unsqueeze(1)
    w2 = w2_mhz.unsqueeze(1)
    s1 = w1 / _SQRT_2LN2
    s2 = w2 / _SQRT_2LN2
    ev = eta.unsqueeze(1)

    L1 = 1.0 / (1.0 + ((f - x1) / w1) ** 2)
    G1 = torch.exp(-0.5 * ((f - x1) / s1) ** 2)
    L2 = 1.0 / (1.0 + ((f - x2) / w2) ** 2)
    G2 = torch.exp(-0.5 * ((f - x2) / s2) ** 2)

    d1 = c1.unsqueeze(1) * (ev * L1 + (1.0 - ev) * G1)
    d2 = c2.unsqueeze(1) * (ev * L2 + (1.0 - ev) * G2)
    return bl * (1.0 - d1 - d2)


def _perturb_freq_axis(freq_ghz, rng, fp):
    f = freq_ghz.copy()
    f += rng.normal(0, fp.shift_mhz_std * 1e-3)
    c = f.mean()
    f = c + (f - c) * (1.0 + rng.normal(0, fp.scale_ppm_std * 1e-6))
    u = np.linspace(-1, 1, len(f))
    f += fp.nonlinear_amp * (u ** 2 - 1.0 / 3.0)
    return f


def generate_single_spectrum(rng, cfg):
    ls = cfg.lineshape
    bl = cfg.baseline
    ns = cfg.noise

    kinds = ["lorentzian", "gaussian", "voigt"]
    kind = rng.choice(kinds, p=ls.lineshape_probs)
    if kind == "lorentzian":
        eta = 1.0
    elif kind == "gaussian":
        eta = 0.0
    else:
        eta = float(rng.uniform(*ls.voigt_eta_range))

    center_ghz = float(rng.uniform(*ls.center_range_ghz))
    w_base = float(rng.uniform(*ls.width_range_mhz))

    aas = min(abs(rng.normal(0, ls.width_asym_sigma)), ls.width_asym_max)
    if rng.random() < 0.5:
        w1, w2 = w_base, w_base * (1.0 - aas)
    else:
        w1, w2 = w_base * (1.0 - aas), w_base

    c_base = float(rng.uniform(*ls.contrast_range))
    aas_c = min(abs(rng.normal(0, ls.contrast_asym_sigma)), ls.contrast_asym_max)
    if rng.random() < 0.5:
        c1, c2 = c_base, c_base * (1.0 - aas_c)
    else:
        c1, c2 = c_base * (1.0 - aas_c), c_base

    s_min = ls.min_split_width_factor * (w1 + w2) / 2.0
    split_mhz = float(max(rng.uniform(*ls.split_range_mhz), s_min))

    bl_off = float(rng.uniform(*bl.offset_range))
    bl_sl = float(rng.uniform(*bl.slope_range))
    bl_qu = float(rng.uniform(*bl.quad_range))

    freq_pert = _perturb_freq_axis(cfg.freq_axis_ghz, rng, cfg.freq_perturb)
    ideal = reconstruct_spectrum_np(
        freq_pert, center_ghz, split_mhz, w1, w2, c1, c2, eta,
        bl_off, bl_sl, bl_qu)

    if bl.slow_bg_amplitude > 0:
        t = np.linspace(0, 1, len(ideal))
        n_per = rng.uniform(0.2, bl.slow_bg_max_periods)
        phase = rng.uniform(0, 2 * np.pi)
        ideal *= (1.0 + bl.slow_bg_amplitude
                  * np.sin(2 * np.pi * n_per * t + phase))


    ph_lo, ph_hi = ns.photon_count_range
    if ph_hi / max(ph_lo, 1) > 30:
        total_ph = int(np.exp(rng.uniform(np.log(max(ph_lo, 1)), np.log(ph_hi))))
    else:
        total_ph = int(rng.integers(ph_lo, ph_hi))
    mult = 1.0 + rng.normal(0, ns.multiplicative_noise_std, size=len(ideal))
    signal = np.clip(ideal * mult * total_ph, 1.0, None)
    counts = rng.poisson(lam=signal).astype(np.float64)
    counts += rng.normal(0, ns.readout_noise_std * total_ph, size=len(counts))

    if ns.bad_pixel_prob > 0:
        bad = rng.random(len(counts)) < ns.bad_pixel_prob
        if bad.any():
            counts[bad] *= rng.uniform(0.1, ns.bad_pixel_scale, int(bad.sum()))

    snr = max(c1, c2) * np.sqrt(total_ph)
    mu, sig = counts.mean(), counts.std()
    if sig < 1e-12:
        sig = 1.0
    normed = ((counts - mu) / sig).astype(np.float32)

    labels = np.array([center_ghz, split_mhz, w1, w2, c1, c2, eta,
                       bl_off, bl_sl, bl_qu], dtype=np.float32)
    meta = {"kind": kind, "photons": total_ph}
    return normed, labels, snr, meta


def generate_single_spectrum_clean(rng, cfg):
    ls = cfg.lineshape
    bl = cfg.baseline

    kinds = ["lorentzian", "gaussian", "voigt"]
    kind = rng.choice(kinds, p=ls.lineshape_probs)
    if kind == "lorentzian":
        eta = 1.0
    elif kind == "gaussian":
        eta = 0.0
    else:
        eta = float(rng.uniform(*ls.voigt_eta_range))

    center_ghz = float(rng.uniform(*ls.center_range_ghz))
    w_base = float(rng.uniform(*ls.width_range_mhz))
    aas = min(abs(rng.normal(0, ls.width_asym_sigma)), ls.width_asym_max)
    if rng.random() < 0.5:
        w1, w2 = w_base, w_base * (1.0 - aas)
    else:
        w1, w2 = w_base * (1.0 - aas), w_base

    c_base = float(rng.uniform(*ls.contrast_range))
    aas_c = min(abs(rng.normal(0, ls.contrast_asym_sigma)), ls.contrast_asym_max)
    if rng.random() < 0.5:
        c1, c2 = c_base, c_base * (1.0 - aas_c)
    else:
        c1, c2 = c_base * (1.0 - aas_c), c_base

    s_min = ls.min_split_width_factor * (w1 + w2) / 2.0
    split_mhz = float(max(rng.uniform(*ls.split_range_mhz), s_min))
    bl_off = float(rng.uniform(*bl.offset_range))
    bl_sl = float(rng.uniform(*bl.slope_range))
    bl_qu = float(rng.uniform(*bl.quad_range))

    freq = cfg.freq_axis_ghz
    ideal = reconstruct_spectrum_np(
        freq, center_ghz, split_mhz, w1, w2, c1, c2, eta,
        bl_off, bl_sl, bl_qu)

    total_ph = 100_000_000
    signal = np.clip(ideal * total_ph, 1.0, None)
    counts = rng.poisson(lam=signal).astype(np.float64)

    snr = max(c1, c2) * np.sqrt(total_ph)
    mu, sig = counts.mean(), counts.std()
    if sig < 1e-12:
        sig = 1.0
    normed = ((counts - mu) / sig).astype(np.float32)

    labels = np.array([center_ghz, split_mhz, w1, w2, c1, c2, eta,
                       bl_off, bl_sl, bl_qu], dtype=np.float32)
    return normed, labels, ideal


class SyntheticODMRDataset(IterableDataset):
    def __init__(self, cfg=CFG, base_seed=42):
        self.cfg = cfg
        self.base_seed = int(base_seed)
        self._iter_count = 0

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi else 0
        self._iter_count += 1
        torch_seed = int(torch.initial_seed()) & 0xFFFFFFFF
        seed_seq = np.random.SeedSequence([
            self.base_seed,
            torch_seed,
            int(worker_id),
            int(self._iter_count),
        ])
        rng = np.random.default_rng(seed_seq)
        while True:
            spec, lab, _, _ = generate_single_spectrum(rng, self.cfg)
            yield (torch.from_numpy(spec).unsqueeze(0),
                   torch.from_numpy(lab))


class FiniteSyntheticDataset(Dataset):
    def __init__(self, n, cfg=CFG, seed=12345):
        rng = np.random.default_rng(seed)
        sp, lb, sn = [], [], []
        for _ in range(n):
            s, l, snr, _ = generate_single_spectrum(rng, cfg)
            sp.append(s)
            lb.append(l)
            sn.append(snr)
        self.spectra = np.stack(sp)
        self.labels = np.stack(lb)
        self.snrs = np.array(sn)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, i):
        return (torch.from_numpy(self.spectra[i]).unsqueeze(0),
                torch.from_numpy(self.labels[i]))


def normalize_odmr_stack(stack, expected_n_freq=None, reduce_repeats="mean",
                         strict=True, return_info=False):
    if expected_n_freq is None:
        expected_n_freq = CFG.n_freq

    arr = np.asarray(stack)
    if arr.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D ODMR stack, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValueError(f"Stack has {n_bad} non-finite values")

    info = {
        "original_shape": tuple(arr.shape),
        "freq_axis": None,
        "repeat_axis": None,
        "reduce_method": None,
        "normalized_shape": None,
    }

    freq_candidates = [ax for ax, sz in enumerate(arr.shape) if sz == expected_n_freq]
    if not freq_candidates:
        raise ValueError(
            f"Could not identify frequency axis in shape {arr.shape}: "
            f"no dimension equals expected_n_freq={expected_n_freq}")

    if arr.ndim == 3:
        freq_axis = freq_candidates[-1] if arr.shape[-1] == expected_n_freq else freq_candidates[0]
        out = np.moveaxis(arr, freq_axis, 0).astype(np.float64, copy=False)
        info["freq_axis"] = freq_axis
        info["normalized_shape"] = tuple(out.shape)
        return (out, info) if return_info else out


    freq_axis = None
    repeat_axis = None
    best_score = None
    axes = list(range(4))
    for f_ax in freq_candidates:
        remaining = [ax for ax in axes if ax != f_ax]
        for r_ax in remaining:
            spatial = [ax for ax in remaining if ax != r_ax]
            if len(spatial) != 2:
                continue
            score = (
                abs(arr.shape[spatial[0]] - arr.shape[spatial[1]]),
                0 if abs(r_ax - f_ax) == 1 else 1,
                -min(arr.shape[spatial[0]], arr.shape[spatial[1]]),
                arr.shape[r_ax],
            )
            if best_score is None or score < best_score:
                best_score = score
                freq_axis = f_ax
                repeat_axis = r_ax

    if freq_axis is None or repeat_axis is None:
        raise ValueError(f"Could not resolve frequency/repeat axes from shape {arr.shape}")

    spatial_axes = [ax for ax in axes if ax not in (freq_axis, repeat_axis)]
    arr = np.transpose(arr, [freq_axis, repeat_axis] + spatial_axes)

    if reduce_repeats == "mean":
        out = arr.mean(axis=1)
    elif reduce_repeats == "median":
        out = np.median(arr, axis=1)
    else:
        raise ValueError(f"Unsupported repeat reduction: {reduce_repeats}")

    out = out.astype(np.float64, copy=False)
    info["freq_axis"] = freq_axis
    info["repeat_axis"] = repeat_axis
    info["reduce_method"] = reduce_repeats
    info["normalized_shape"] = tuple(out.shape)
    return (out, info) if return_info else out


def load_odmr_stack(path, expected_n_freq=None, reduce_repeats="mean",
                    strict=True, verbose=True):
    if path.endswith(".npy"):
        stk = np.load(path)
    elif path.endswith((".tif", ".tiff")):
        from tifffile import imread
        stk = imread(path)
    else:
        raise ValueError(f"Unsupported format: {path}")

    normed, info = normalize_odmr_stack(
        stk,
        expected_n_freq=expected_n_freq,
        reduce_repeats=reduce_repeats,
        strict=strict,
        return_info=True,
    )

    if verbose:
        print(f"[stack] loaded: {path}")
        print(f"[stack] original shape:    {info['original_shape']}")
        print(f"[stack] frequency axis:   {info['freq_axis']}")
        if info['repeat_axis'] is not None:
            print(f"[stack] repeat axis:      {info['repeat_axis']}")
            print(f"[stack] repeat reduction: {info['reduce_method']}")
        print(f"[stack] normalized shape: {info['normalized_shape']}")
    return normed


def validate_stack_pair(s1, s2, nf):
    n1, h1, w1 = s1.shape
    n2, h2, w2 = s2.shape
    if n1 != nf:
        raise ValueError(f"Main stack: expected {nf} frames, got {n1}")
    if n2 != nf:
        raise ValueError(f"ZF stack: expected {nf} frames, got {n2}")
    if (h1, w1) != (h2, w2):
        raise ValueError(f"Shape mismatch: ({h1},{w1}) vs ({h2},{w2})")


def stack_to_pixel_spectra(stk):
    Nf, H, W = stk.shape
    return stk.reshape(Nf, -1).T.copy(), H, W


def zscore_spectra(sp):
    mu = sp.mean(axis=1, keepdims=True)
    sig = sp.std(axis=1, keepdims=True)
    sig[sig < 1e-12] = 1.0
    return ((sp - mu) / sig).astype(np.float32)


def generate_synthetic_stack(cfg=CFG, H=64, W=64, seed=2026, save_dir=None):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    freq = cfg.freq_axis_ghz
    Nf = cfg.n_freq
    smooth = min(H, W) / 8.0

    def _smooth_field(base, amp, lo, hi):
        raw = rng.standard_normal((H, W))
        sm = gaussian_filter(raw, sigma=smooth)
        sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-12)
        return np.clip(base + amp * (sm - 0.5), lo, hi).astype(np.float64)

    center_map = _smooth_field(2.870, 0.010, 2.855, 2.885)
    split_map = _smooth_field(10.0, 12.0, 2.0, 25.0)
    w1_map = _smooth_field(5.0, 4.0, 2.0, 9.0)

    w2_base = gaussian_filter(rng.standard_normal((H, W)), sigma=smooth)
    w2_base = (w2_base - w2_base.min()) / (w2_base.max() - w2_base.min() + 1e-12)
    w2_map = np.clip(w1_map + 0.8 * (w2_base - 0.5), 2.0, 9.0).astype(np.float64)

    c1_map = _smooth_field(0.06, 0.08, 0.01, 0.15)
    c2_base = gaussian_filter(rng.standard_normal((H, W)), sigma=smooth)
    c2_base = (c2_base - c2_base.min()) / (c2_base.max() - c2_base.min() + 1e-12)
    c2_map = np.clip(c1_map + 0.03 * (c2_base - 0.5), 0.01, 0.15).astype(np.float64)

    eta_map = _smooth_field(0.65, 0.5, 0.1, 1.0)
    bl_off_map = _smooth_field(1.0, 0.04, 0.95, 1.05)
    bl_sl_map = _smooth_field(0.0, 0.03, -0.02, 0.02)
    bl_qu_map = _smooth_field(0.0, 0.01, -0.01, 0.01)


    r1, c1_z = H // 4, W // 4
    sz = max(H, W) // 8
    y_lo, y_hi = max(r1 - sz, 0), min(r1 + sz, H)
    x_lo, x_hi = max(c1_z - sz, 0), min(c1_z + sz, W)
    c1_map[y_lo:y_hi, x_lo:x_hi] *= 0.3
    c2_map[y_lo:y_hi, x_lo:x_hi] *= 0.3

    r2, c2_z = 3 * H // 4, 3 * W // 4
    y_lo2, y_hi2 = max(r2 - sz, 0), min(r2 + sz, H)
    x_lo2, x_hi2 = max(c2_z - sz, 0), min(c2_z + sz, W)
    bl_sl_map[y_lo2:y_hi2, x_lo2:x_hi2] *= 3.0
    bl_qu_map[y_lo2:y_hi2, x_lo2:x_hi2] *= 3.0

    photon_base = 1_000_000
    photons_map = np.full((H, W), photon_base, dtype=np.float64)

    strip_lo = H // 2 - max(H // 16, 1)
    strip_hi = H // 2 + max(H // 16, 1)
    photons_map[strip_lo:strip_hi, :] *= 0.1

    gamma_e = cfg.gamma_e_mhz_per_gauss
    bz_map = split_map / (2.0 * gamma_e)


    stack = np.zeros((Nf, H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            ideal = reconstruct_spectrum_np(
                freq, center_map[i, j], split_map[i, j],
                w1_map[i, j], w2_map[i, j],
                c1_map[i, j], c2_map[i, j],
                float(eta_map[i, j]),
                float(bl_off_map[i, j]), float(bl_sl_map[i, j]),
                float(bl_qu_map[i, j]))
            ph = photons_map[i, j]
            stack[:, i, j] = rng.poisson(lam=np.clip(ideal * ph, 1.0, None))


    c_max = np.maximum(c1_map, c2_map)
    snr_per_pixel = c_max * np.sqrt(photons_map)

    result = {
        "stack": stack, "H": H, "W": W,


        "center_ghz_gt": center_map,
        "split_mhz_gt": split_map,
        "w1_mhz_gt": w1_map,
        "w2_mhz_gt": w2_map,
        "c1_gt": c1_map,
        "c2_gt": c2_map,
        "eta_gt": eta_map,
        "bl_offset_gt": bl_off_map,
        "bl_slope_gt": bl_sl_map,
        "bl_quad_gt": bl_qu_map,
        "bz_gauss_gt": bz_map,


        "photons_per_pixel": photons_map,
        "snr_per_pixel": snr_per_pixel,

        "noise_map": photons_map,


        "center_map": center_map,
        "split_map": split_map,
        "width_map": (w1_map + w2_map) / 2.0,
        "contrast_map": (c1_map + c2_map) / 2.0,
        "bl_offset_map": bl_off_map,
        "bl_slope_map": bl_sl_map,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "synthetic_stack.npy"), stack)
        for k in [k for k in result if k.endswith("_gt") or k in
                  ("photons_per_pixel", "snr_per_pixel")]:
            np.save(os.path.join(save_dir, f"{k}.npy"), result[k])
        print(f"[odmr_core] Saved synthetic stack {Nf}x{H}x{W} to {save_dir}/")
    return result


def splitting_to_bz(delta_mhz, delta0_mhz=0.0,
                    gamma_e=CFG.gamma_e_mhz_per_gauss):
    hd = np.asarray(delta_mhz) / (2.0 * gamma_e)
    hd0 = np.asarray(delta0_mhz) / (2.0 * gamma_e)
    arg = hd ** 2 - hd0 ** 2
    valid = arg >= 0
    return np.sqrt(np.where(valid, arg, 0.0)), valid


def params_to_maps(pred, cfg=CFG):
    return {
        "center_ghz": pred[..., 0],
        "split_mhz": pred[..., 1],
        "w1_mhz": pred[..., 2],
        "w2_mhz": pred[..., 3],
        "c1": pred[..., 4],
        "c2": pred[..., 5],
        "eta": pred[..., 6],
        "bl_offset": pred[..., 7],
        "bl_slope": pred[..., 8],
        "bl_quad": pred[..., 9],
        "x1_ghz": pred[..., 0] - pred[..., 1] / 2e3,
        "x2_ghz": pred[..., 0] + pred[..., 1] / 2e3,
    }


def full_physics_pipeline(pred_field, pred_zf=None, delta0_mhz=0.0,
                          gamma_e=CFG.gamma_e_mhz_per_gauss, cfg=CFG):
    maps = params_to_maps(pred_field, cfg)
    sp = maps["split_mhz"]
    d0 = params_to_maps(pred_zf, cfg)["split_mhz"] if pred_zf is not None else delta0_mhz
    bz, valid = splitting_to_bz(sp, d0, gamma_e)
    inv_frac = 1.0 - valid.mean()
    if inv_frac > 0:
        print(f"[physics] {inv_frac*100:.1f}% invalid Bz pixels")
    return {
        "splitting_mhz": sp,
        "bz_gauss": bz,
        "valid_mask": valid,
        **maps
    }
