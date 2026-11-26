import numpy as np
import torch

from pipeline import load_checkpoint, predict_spectra
from sparse_fast import FULL_N, SparseFreqAxis, make_sparse_cfg, sparse_indices
from odmr_core import zscore_spectra


PARAMETER_NAMES = (
    "center_ghz", "split_mhz", "w1_mhz", "w2_mhz", "c1", "c2",
    "eta", "bl_offset", "bl_slope", "bl_quad",
)


def load_model(weight_file):
    """Load the included model on CPU."""
    model, _ = load_checkpoint(str(weight_file), make_sparse_cfg(200), device=torch.device("cpu"))
    return model


def infer_spectra(model, frequency_ghz, fluorescence):
    axis = np.asarray(frequency_ghz, dtype=np.float64)
    values = np.asarray(fluorescence, dtype=np.float32)
    full_grid = SparseFreqAxis(model.cfg.n_freq)._full_axis
    k = model.cfg.n_freq
    subset = sparse_indices(k)
    if values.ndim != 2 or axis.ndim != 1 or not len(values) or values.shape[1] != axis.size:
        raise ValueError("Expected a frequency vector and a nonempty spectra-by-frequency array.")
    if not np.isfinite(axis).all() or not np.isfinite(values).all():
        raise ValueError("Frequencies and fluorescence must be finite.")
    if axis.size == FULL_N:
        expected, indices = full_grid, subset
    elif axis.size == k:
        expected, indices = full_grid[subset], slice(None)
    else:
        raise ValueError("Expected the 500-point grid or the checkpoint's fixed frequency subset.")
    if not np.allclose(axis, expected, rtol=0, atol=5e-8):
        raise ValueError("Frequency grid does not match the checkpoint.")

    normalized = zscore_spectra(values[:, indices])
    return predict_spectra(model, normalized, torch.device("cpu"), return_sigma=True)
