import argparse
from pathlib import Path

import pandas as pd

from review_ranking import internal_review_mask

def main():
    parser = argparse.ArgumentParser(description="Rank tasks from precomputed PTRA scores.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv)
    try:
        required = {"k", "triplet_id", "risk_residual", "risk_domain", "risk_combined", "risk_rho", "raw_residual"}
        if frame.empty or not required.issubset(frame.columns) or frame[["k", "triplet_id"]].isna().any().any():
            raise ValueError("Expected a nonempty task table with all documented columns and valid keys.")
        flags = internal_review_mask(frame, "PTRA")
    except ValueError as exc:
        parser.error(str(exc))
    frame[["k", "triplet_id"]].assign(priority_review=flags).to_csv(args.output_csv, index=False)
    print(f"Selected {flags.sum()} of {len(frame)} task/configuration pairs.")


if __name__ == "__main__":
    main()
