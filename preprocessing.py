"""CSV adapter and repeat-sweep preprocessing for selected spectra."""

import csv
from dataclasses import dataclass

import numpy as np


def sigma_from_stdrep(sd: np.ndarray, n_groups: int,
                      floor_frac: float = 0.01) -> np.ndarray:
    sd = np.asarray(sd, dtype=np.float64)
    sig = sd / np.sqrt(max(n_groups - 1, 1))
    med = np.median(sig, axis=-1, keepdims=True)
    return np.maximum(sig, np.maximum(med * floor_frac, 1e-9))


@dataclass(frozen=True)
class MeanSpectrum:
    frequency_ghz: np.ndarray
    mean_counts: np.ndarray
    sem_counts: np.ndarray
    repeat_count: int


def load_repeated_counts(path):
    """Load frequency_ghz and consecutive repeat_01, repeat_02, ... columns."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
        expected = ["frequency_ghz"] + [
            f"repeat_{index:02d}" for index in range(1, len(header))
        ]
        if len(header) < 3 or header != expected:
            raise ValueError("Expected frequency_ghz followed by at least two numbered repeats.")
        data = np.loadtxt(stream, delimiter=",", ndmin=2)
    if data.shape != (500, len(header)):
        raise ValueError("Expected 500 frequency rows and one value per column.")
    if not np.isfinite(data).all() or (data[:, 1:] < 0).any():
        raise ValueError("Frequencies and nonnegative camera counts must be finite.")
    return data[:, 0], data[:, 1:]


def prepare_mean_spectrum(frequency_ghz, counts):
    """Average sweeps and estimate the SEM with the original fractional noise floor."""
    n = counts.shape[1]
    mean = np.mean(counts, axis=1, dtype=np.float64).astype(np.float32)
    population_sd = np.std(counts, axis=1, ddof=0, dtype=np.float64)
    sem = sigma_from_stdrep(population_sd, n).astype(np.float32)
    return MeanSpectrum(frequency_ghz, mean, sem, n)
