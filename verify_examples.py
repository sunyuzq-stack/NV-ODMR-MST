"""Check the public examples and fixed-grid handling."""

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_repeated_spectrum import analyze
from submission_inference import infer_spectra, load_model
from review_ranking import internal_review_mask
from sparse_fast import sparse_indices

ROOT = Path(__file__).resolve().parent


def main():
    axis, fluorescence = np.loadtxt(
        ROOT / "examples/synthetic_odmr_spectrum.csv", delimiter=",", skiprows=1
    ).T
    model = load_model(ROOT / "weights/odmr_cnn_k200_weights.pt")
    mean, sd = infer_spectra(model, axis, fluorescence[None, :])
    subset = sparse_indices(200)
    subset_mean, subset_sd = infer_spectra(model, axis[subset], fluorescence[None, subset])
    np.testing.assert_allclose(mean, subset_mean, rtol=0, atol=1e-6)
    np.testing.assert_allclose(sd, subset_sd, rtol=0, atol=1e-6)
    np.testing.assert_allclose([mean[0, 1], sd[0, 1]], [9.959386, 0.080951], rtol=0, atol=2e-5)
    table = pd.read_csv(ROOT / "examples/precomputed_risk_scores.csv")
    np.testing.assert_array_equal(
        table.loc[internal_review_mask(table, "PTRA"), "triplet_id"].to_numpy(), ["example_4"]
    )
    for name, expected in (
        ("synthetic_repeated_counts.csv", [11.557155, 2.799922, 1.135891]),
        ("measured_spiral_repeated_counts.csv", [11.652093, 1.116524, 1.188422]),
    ):
        example = ROOT / "examples" / name
        if not example.exists() and name.startswith("measured_"):
            continue
        prepared, result = analyze(example, ROOT / "weights/odmr_cnn_k200_weights.pt")
        assert prepared.repeat_count == 10 and result["input_frequency_count"] == 200
        np.testing.assert_allclose([
            result["parameter_estimates"]["split_mhz"],
            result["doublet_resolvability"],
            result["noise_weighted_forward_residual"],
        ], expected, rtol=0, atol=3e-5)
    print("Inference, repeated-scan preprocessing, diagnostics, and review ranking passed.")


if __name__ == "__main__":
    main()
