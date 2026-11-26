import argparse
from pathlib import Path

import numpy as np

from submission_inference import infer_spectra, load_model

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Estimate ODMR doublet splitting.")
    parser.add_argument("--input", type=Path, default=ROOT / "examples/synthetic_odmr_spectrum.csv")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/odmr_cnn_k200_weights.pt")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split(",")
        if header != ["frequency_ghz", "fluorescence"]:
            parser.error("Expected CSV columns: frequency_ghz,fluorescence.")
        data = np.loadtxt(stream, delimiter=",", ndmin=2)
    if data.shape[1] != 2:
        parser.error("The spectrum CSV must have two columns: frequency_ghz,fluorescence.")
    estimates, sd = infer_spectra(load_model(args.weights), data[:, 0], data[:, 1][None, :])
    print(f"Estimated splitting: {estimates[0, 1]:.6f} MHz")
    print(f"Predicted splitting SD: {sd[0, 1]:.6f} MHz")


if __name__ == "__main__":
    main()
