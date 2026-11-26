import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from review_ranking import internal_review_mask

def main():
    parser = argparse.ArgumentParser(description="Rank tasks from precomputed PTRA scores.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    try:
        if args.input_csv.resolve() == args.output_csv.resolve() or (
            args.output_csv.exists() and args.input_csv.samefile(args.output_csv)
        ):
            raise ValueError("Choose an output path different from the input CSV.")
        frame = pd.read_csv(args.input_csv)
        scoring_columns = ["risk_residual", "risk_domain", "risk_combined", "risk_rho", "raw_residual"]
        required = {"k", "triplet_id", *scoring_columns}
        if frame.empty or not required.issubset(frame.columns) or frame[["k", "triplet_id"]].isna().any().any():
            raise ValueError("Expected a nonempty task table with all documented columns and valid keys.")
        frequencies = pd.to_numeric(frame["k"], errors="raise").to_numpy(dtype=float)
        if not (np.isfinite(frequencies) & (frequencies > 0) & (frequencies == np.floor(frequencies))).all():
            raise ValueError("Frequency counts in column k must be positive integers.")
        for column in scoring_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        flags = internal_review_mask(frame, "PTRA")
        frame[["k", "triplet_id"]].assign(priority_review=flags).to_csv(args.output_csv, index=False)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Selected {flags.sum()} of {len(frame)} task/configuration pairs.")


if __name__ == "__main__":
    main()
