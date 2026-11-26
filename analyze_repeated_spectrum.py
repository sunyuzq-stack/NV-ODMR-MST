"""Run the measured-count preprocessing, inference and spectral diagnostics example."""

import argparse
import json
from pathlib import Path

import numpy as np

from submission_inference import PARAMETER_NAMES, infer_spectra, load_model
from preprocessing import load_repeated_counts, prepare_mean_spectrum
from sparse_fast import sparse_indices
from spectral_diagnostics import doublet_resolvability, weighted_affine_residual

ROOT = Path(__file__).resolve().parent


def analyze(input_csv, weights):
    spectrum = prepare_mean_spectrum(*load_repeated_counts(input_csv))
    model = load_model(weights)
    estimates, predicted_sd = infer_spectra(
        model, spectrum.frequency_ghz, spectrum.mean_counts[None, :]
    )
    index = sparse_indices(model.cfg.n_freq)
    forward = weighted_affine_residual(
        spectrum.frequency_ghz[index],
        spectrum.mean_counts[None, index],
        spectrum.sem_counts[None, index],
        estimates,
    )
    residual = float(forward["score"][0])
    rho = float(doublet_resolvability(estimates)[0])
    return spectrum, {
        "repeat_count": spectrum.repeat_count,
        "input_frequency_count": int(index.size),
        "parameter_estimates": dict(zip(PARAMETER_NAMES, map(float, estimates[0]))),
        "predicted_splitting_sd_mhz": float(predicted_sd[0, 1]),
        "doublet_resolvability": rho if np.isfinite(rho) else None,
        "noise_weighted_forward_residual": residual if np.isfinite(residual) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/odmr_cnn_k200_weights.pt")
    parser.add_argument("--prepared-csv", type=Path, help="Write the 500-point mean and SEM.")
    args = parser.parse_args()
    try:
        spectrum, result = analyze(args.input_csv, args.weights)
    except ValueError as exc:
        parser.error(str(exc))
    if args.prepared_csv:
        if args.prepared_csv.resolve() == args.input_csv.resolve():
            parser.error("Choose an output path different from the input.")
        np.savetxt(
            args.prepared_csv,
            np.column_stack((spectrum.frequency_ghz, spectrum.mean_counts, spectrum.sem_counts)),
            delimiter=",", header="frequency_ghz,mean_counts,sem_counts", comments="", fmt="%.12g",
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
