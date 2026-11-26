import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FreqAxisConfig:
    start_ghz: float = 2.8200
    step_ghz: float = 0.0002
    n_points: int = 500

    @property
    def end_ghz(self) -> float:
        return self.start_ghz + (self.n_points - 1) * self.step_ghz

    @property
    def axis_ghz(self) -> np.ndarray:
        return np.linspace(
            self.start_ghz,
            self.start_ghz + (self.n_points - 1) * self.step_ghz,
            self.n_points
        )

    @property
    def axis_mhz(self) -> np.ndarray:
        return self.axis_ghz * 1e3

    @property
    def span_mhz(self) -> float:
        return (self.end_ghz - self.start_ghz) * 1e3

    def validate(self):
        assert self.n_points > 0
        assert self.step_ghz > 0


@dataclass
class BaselineConfig:
    offset_range: Tuple[float, float] = (0.95, 1.05)
    slope_range: Tuple[float, float] = (-0.02, 0.02)
    quad_range: Tuple[float, float] = (-0.01, 0.01)
    slow_bg_amplitude: float = 0.005
    slow_bg_max_periods: float = 2.0


@dataclass
class NoiseConfig:
    photon_count_range: Tuple[int, int] = (3_000, 3_600_000)
    readout_noise_std: float = 0.002
    multiplicative_noise_std: float = 0.005
    bad_pixel_prob: float = 0.002
    bad_pixel_scale: float = 5.0


@dataclass
class LineshapeConfig:
    center_range_ghz: Tuple[float, float] = (2.855, 2.885)
    split_range_mhz: Tuple[float, float] = (3.0, 22.0)
    width_range_mhz: Tuple[float, float] = (3.0, 12.0)  # Half width at half maximum (MHz).
    contrast_range: Tuple[float, float] = (0.012, 0.15)
    width_asym_sigma: float = 0.083
    width_asym_max: float = 0.25
    contrast_asym_sigma: float = 0.083
    contrast_asym_max: float = 0.25
    min_split_width_factor: float = 0.75  # Legacy training default. The released inference weights use the expanded support.
    lineshape_probs: List[float] = field(default_factory=lambda: [0.6, 0.15, 0.25])  # Lorentzian, Gaussian, and pseudo-Voigt sampling probabilities.
    voigt_eta_range: Tuple[float, float] = (0.3, 0.9)


@dataclass
class FreqPerturbConfig:
    shift_mhz_std: float = 0.3
    scale_ppm_std: float = 200.0
    nonlinear_amp: float = 0.0002


@dataclass
class ModelConfig:
    conv_layers: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (32,  9, 2),    # 500 -> 246
        (64,  7, 2),    # 246 -> 120
        (96,  5, 2),    # 120 -> 58
        (128, 5, 2),    # 58  -> 27
        (128, 3, 2),    # 27  -> 13  (flat = 128*13 = 1664)
    ])
    fc_hidden: List[int] = field(default_factory=lambda: [128])

    n_param_outputs: int = 10
    n_raw_outputs: int = 20  # Ten parameter estimates and ten log variances.
    dropout: float = 0.0
    use_batchnorm: bool = True


SPARSE_ARCH_BY_K = {
    500: ([(48, 9, 2), (96, 7, 2), (160, 5, 2), (224, 5, 2), (256, 3, 2)], [384, 192]),
    400: ([(48, 9, 2), (96, 7, 2), (160, 5, 2), (224, 5, 2), (256, 3, 2)], [384, 192]),
    300: ([(48, 9, 2), (96, 7, 2), (160, 5, 2), (224, 5, 2), (256, 3, 2)], [384, 192]),
    200: ([(48, 7, 2), (96, 5, 2), (160, 5, 2), (224, 3, 2)],              [384, 192]),
    100: ([(48, 5, 2), (128, 3, 2), (192, 3, 2)],                          [384, 192]),
    80:  ([(48, 5, 2), (128, 3, 2), (192, 3, 2)],                          [384, 192]),
}


def sparse_arch_fallback(k):
    if k >= 300:
        return ([(48, 9, 2), (96, 7, 2), (160, 5, 2), (224, 5, 2), (256, 3, 2)], [384, 192])
    elif k >= 150:
        return ([(48, 7, 2), (96, 5, 2), (160, 5, 2), (224, 3, 2)], [384, 192])
    else:
        return ([(48, 5, 2), (128, 3, 2), (192, 3, 2)], [384, 192])


SPARSE_VARIANTS = {
    "K500_match": {
        "k": 500,
        "conv": [(128, 9, 2), (256, 7, 2), (512, 5, 2), (512, 5, 2), (512, 3, 2)],
        "fc":   [1024, 256],
        "train": {
            "batch_size": 2048,
            "lr_max": 2.5e-4,
            "lr_min": 1.0e-5,
            "warmup_fraction": 0.05,   # 5% warmup
            "epochs": 1500,
            "val_batches": 8,
            "use_amp": True,
            "num_workers": 12,
        },
        "target_samples": 2.0e9,
    },
}


def _make_arch_variant(conv, fc=(1024, 256), lr_max=2.5e-4,
                       epochs=1500, target_samples=2.0e9,
                       batch_size=2048, num_workers=12):
    return {
        "k": 500,
        "conv": [tuple(t) for t in conv],
        "fc":   list(fc),
        "train": {
            "batch_size": batch_size,
            "lr_max": lr_max,
            "lr_min": 1.0e-5,
            "warmup_fraction": 0.05,
            "epochs": epochs,
            "val_batches": 8,
            "use_amp": True,
            "num_workers": num_workers,
        },
        "target_samples": target_samples,
    }


_ARCH_ABLATION = {

    "abl_d3_w64":  _make_arch_variant([(64, 9, 2), (128, 7, 2), (256, 5, 2)]),                 # ~8M
    "abl_d3_w96":  _make_arch_variant([(96, 9, 2), (192, 7, 2), (384, 5, 2)]),                 # ~18M

    "abl_d4_w96":  _make_arch_variant([(96, 9, 2), (192, 7, 2), (384, 5, 2), (384, 5, 2)]),    # ~10M
    "abl_d4_w128": _make_arch_variant([(128, 9, 2), (256, 7, 2), (512, 5, 2), (512, 5, 2)]),   # ~16M

    "abl_d5_w128": _make_arch_variant([(128, 9, 2), (256, 7, 2), (512, 5, 2), (512, 5, 2), (512, 3, 2)]),    # ~10M
    "abl_d5_w192": _make_arch_variant([(192, 9, 2), (384, 7, 2), (768, 5, 2), (768, 5, 2), (512, 3, 2)]),    # ~13M
    "abl_d5_w256": _make_arch_variant([(256, 9, 2), (512, 7, 2), (1024, 5, 2), (1024, 5, 2), (512, 3, 2)]),  # ~17M = K500_huge
}
SPARSE_VARIANTS.update(_ARCH_ABLATION)

_LADDER_CONV   = [(96, 9, 2), (192, 7, 2), (384, 5, 2), (384, 5, 2)]
_LADDER_LEVELS = ("A_ideal", "B_voigt", "C_systematic", "D_full")
_LADDER_SEEDS  = (1337, 2024, 7)
_LADDER = {}
for _lv in _LADDER_LEVELS:
    for _sd in _LADDER_SEEDS:
        _v = _make_arch_variant(_LADDER_CONV, target_samples=7.0e8)
        _v["realism"] = _lv
        _v["seed"] = _sd
        _LADDER[f"lad_{_lv}_s{_sd}"] = _v

_v = _make_arch_variant(_LADDER_CONV, target_samples=7.0e8)
_v["realism"] = "D_full"
_v["seed"] = 1337
_v["cfg_overrides"] = {"freq_perturb.shift_mhz_std": 0.05}
_LADDER["lad_D_lowshift_s1337"] = _v

SPARSE_VARIANTS.update(_LADDER)


_v = _make_arch_variant(_LADDER_CONV, target_samples=7.0e8)
_v["realism"] = "D_full"
_v["seed"] = 1337
SPARSE_VARIANTS["lad_E_calibrated_s1337"] = _v

_v = _make_arch_variant(_LADDER_CONV, target_samples=7.0e8)
_v["realism"] = "D_full"
_v["seed"] = 2024
SPARSE_VARIANTS["lad_D_full_s2024"] = _v


@dataclass
class LossConfig:
    nll_weights: List[float] = field(default_factory=lambda: [
        2.0,
        2.0,
        1.0,   # w1
        1.0,   # w2
        0.0,   # c1
        0.0,   # c2
        0.3,   # eta
        0.0,
        0.1,   # bl_slope
        0.1,   # bl_quad
    ])

    log_var_min: float = -16.0
    log_var_max_per_param: List[float] = field(default_factory=lambda: [
        -4.0,
         7.0,
         7.0,
         7.0,   # w2_mhz
        -2.0,
        -2.0,   # c2
         1.0,
        -1.0,
        -4.0,
        -6.0,
    ])

    w_spectral: float = 0.1
    enable_spectral_loss: bool = True


@dataclass
class TrainConfig:
    epochs: int = 300
    batch_size: int = 128
    batches_per_epoch: int = 256
    val_batches: int = 32
    lr_max: float = 1.0e-4
    lr_min: float = 1.0e-6
    warmup_fraction: float = 0.10
    weight_decay: float = 1.0e-2
    betas: Tuple[float, float] = (0.9, 0.95)
    use_amp: bool = False
    grad_clip: float = 0.5
    num_workers: int = 2


@dataclass
class BenchmarkConfig:
    n_pixels: int = 500
    high_info_percentile: float = 0.70
    worst_k: int = 20
    fit_max_iter: int = 2000
    n_mc_iter: int = 200
    enable_mc: bool = True
    enable_hybrid: bool = True


@dataclass
class ConfidenceConfig:
    method: str = "quantile_exp"
    alpha: float = 2.0
    quantile_lo: float = 0.05
    quantile_hi: float = 0.95


@dataclass
class EvalConfig:
    snr_bins: List[Tuple[float, float, str]] = field(default_factory=lambda: [
        (0.0,  2.0,   "snr_lt2"),
        (2.0,  5.0,   "snr_2_5"),
        (5.0,  10.0,  "snr_5_10"),
        (10.0, 30.0,  "snr_10_30"),
        (30.0, 100.0, "snr_30_100"),
        (100.0, 1e9,  "snr_gt100"),
    ])


@dataclass
class Config:
    seed: int = 42
    run_name: str = "default"
    project_dir: str = os.path.dirname(os.path.abspath(__file__))
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    best_model_name: str = "best_model.pt"
    device: str = "cuda"

    freq: FreqAxisConfig = field(default_factory=FreqAxisConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    lineshape: LineshapeConfig = field(default_factory=LineshapeConfig)
    freq_perturb: FreqPerturbConfig = field(default_factory=FreqPerturbConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    infer_batch_size: int = 512
    gamma_e_mhz_per_gauss: float = 2.8025
    test_samples: int = 5000

    @property
    def n_freq(self) -> int:
        return self.freq.n_points

    @property
    def freq_axis_ghz(self) -> np.ndarray:
        return self.freq.axis_ghz

    @property
    def freq_axis_mhz(self) -> np.ndarray:
        return self.freq.axis_mhz

    def validate(self):
        self.freq.validate()

    def ensure_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


    _CAL_MAP = {
        "photon_count_range": ("noise", "photon_count_range"),
        "contrast_range": ("lineshape", "contrast_range"),
        "width_range_mhz": ("lineshape", "width_range_mhz"),
        "baseline_offset_range": ("baseline", "offset_range"),
        "baseline_slope_range": ("baseline", "slope_range"),
        "baseline_quad_range": ("baseline", "quad_range"),
        "center_range_ghz": ("lineshape", "center_range_ghz"),
        "split_range_mhz": ("lineshape", "split_range_mhz"),
    }


    _CAL_SCALARS = {
        "multiplicative_noise_std": ("noise", "multiplicative_noise_std"),
        "readout_noise_std":        ("noise", "readout_noise_std"),
        "bad_pixel_prob":           ("noise", "bad_pixel_prob"),
        "bad_pixel_scale":          ("noise", "bad_pixel_scale"),
        "slow_bg_amplitude":        ("baseline", "slow_bg_amplitude"),
    }

    @staticmethod
    def _validate_range(key, val):
        if not (isinstance(val, (list, tuple)) and len(val) == 2):
            raise ValueError(f"Calibration '{key}': expected [lo, hi], got {val}")
        lo, hi = val
        if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
            raise ValueError(f"Calibration '{key}': non-numeric bounds {val}")
        if lo > hi:
            raise ValueError(f"Calibration '{key}': lo={lo} > hi={hi}")

    def apply_calibration(self, cal_path: str):
        with open(cal_path, "r", encoding="utf-8-sig") as f:
            cal = json.load(f)
        applied, skipped = [], []


        for key, (sub, attr) in self._CAL_MAP.items():
            if key not in cal:
                continue
            val = cal[key]
            if val is None or val == "unavailable":
                skipped.append(key)
                continue
            try:
                self._validate_range(key, val)
            except ValueError as e:
                print(f"[config] WARNING: {e} — skipping")
                skipped.append(key)
                continue
            setattr(getattr(self, sub), attr, tuple(val))
            applied.append(key)


        for key, (sub, attr) in self._CAL_SCALARS.items():
            if key not in cal:
                continue
            val = cal[key]
            if val is None or val == "unavailable":
                skipped.append(key)
                continue
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                print(f"[config] WARNING: '{key}' 应为标量，得到 {val!r} — 跳过")
                skipped.append(key)
                continue
            if val < 0:
                print(f"[config] WARNING: '{key}' 为负 ({val}) — 跳过")
                skipped.append(key)
                continue
            setattr(getattr(self, sub), attr, float(val))
            applied.append(key)

        print(f"[config] Applied calibration from {cal_path}: {applied}")
        if skipped:
            print(f"[config]   Skipped (unavailable/invalid): {skipped}")

    REALISM_LEVELS = ("A_ideal", "B_voigt", "C_systematic", "D_full")

    def apply_realism_level(self, level: str):
        if level not in self.REALISM_LEVELS:
            raise ValueError(f"未知真实度档 {level!r}；可选 {self.REALISM_LEVELS}")
        ls, bl = self.lineshape, self.baseline
        fp, ns = self.freq_perturb, self.noise

        snap = dict(
            probs=list(ls.lineshape_probs),
            off=tuple(bl.offset_range), sl=tuple(bl.slope_range),
            qu=tuple(bl.quad_range), bg=float(bl.slow_bg_amplitude),
            shift=float(fp.shift_mhz_std), scale=float(fp.scale_ppm_std),
            nl=float(fp.nonlinear_amp),
            mult=float(ns.multiplicative_noise_std),
            rd=float(ns.readout_noise_std), badp=float(ns.bad_pixel_prob),
        )

        ls.lineshape_probs = [1.0, 0.0, 0.0]
        bl.offset_range = (1.0, 1.0)
        bl.slope_range = (0.0, 0.0)
        bl.quad_range = (0.0, 0.0)
        bl.slow_bg_amplitude = 0.0
        fp.shift_mhz_std = 0.0
        fp.scale_ppm_std = 0.0
        fp.nonlinear_amp = 0.0
        ns.multiplicative_noise_std = 0.0
        ns.readout_noise_std = 0.0
        ns.bad_pixel_prob = 0.0

        if level in ("B_voigt", "C_systematic", "D_full"):
            ls.lineshape_probs = snap["probs"]
        if level in ("C_systematic", "D_full"):
            bl.offset_range = snap["off"]
            bl.slope_range = snap["sl"]
            bl.quad_range = snap["qu"]
            bl.slow_bg_amplitude = snap["bg"]
            fp.shift_mhz_std = snap["shift"]
            fp.scale_ppm_std = snap["scale"]
            fp.nonlinear_amp = snap["nl"]
        if level == "D_full":
            ns.multiplicative_noise_std = snap["mult"]
            ns.readout_noise_std = snap["rd"]
            ns.bad_pixel_prob = snap["badp"]

        print(f"[config] realism={level}  probs={ls.lineshape_probs}  "
              f"bl_off={bl.offset_range}  shift={fp.shift_mhz_std}  "
              f"scale={fp.scale_ppm_std}  nl={fp.nonlinear_amp}  "
              f"mult={ns.multiplicative_noise_std}  "
              f"readout={ns.readout_noise_std}  badpix={ns.bad_pixel_prob}")


CFG = Config()
CFG.validate()
