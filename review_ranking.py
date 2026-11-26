import numpy as np
import pandas as pd

REVIEW_FRACTION = 0.20

def stable_ids(frame: pd.DataFrame, base: str) -> np.ndarray:
    return (
        frame[base].astype(str) + "#" + frame.index.astype(str)
    ).to_numpy(dtype=str)


def internal_low_to_high_order(frame: pd.DataFrame, method: str) -> np.ndarray:
    ids = stable_ids(frame, "triplet_id")
    if method == "PTRA":
        veto = (
            (frame["risk_residual"].to_numpy(float) >= 1.0)
            | (frame["risk_domain"].to_numpy(float) >= 1.0)
        ).astype(int)
        combined = np.nan_to_num(frame["risk_combined"].to_numpy(float), nan=1.0)
        rho_risk = np.nan_to_num(frame["risk_rho"].to_numpy(float), nan=1.0)
        residual = np.nan_to_num(frame["raw_residual"].to_numpy(float), nan=np.inf)
        return np.lexsort((ids, residual, rho_risk, combined, veto))
    if method == "rho":
        score = np.nan_to_num(frame["risk_rho"].to_numpy(float), nan=1.0)
        rho = np.nan_to_num(frame["raw_rho"].to_numpy(float), nan=-np.inf)
        return np.lexsort((ids, -rho, score))
    if method == "variance":
        score = np.nan_to_num(frame["risk_sigma"].to_numpy(float), nan=1.0)
        raw = np.nan_to_num(frame["raw_sigma"].to_numpy(float), nan=np.inf)
        return np.lexsort((ids, raw, score))
    if method == "residual":
        score = np.nan_to_num(frame["risk_residual"].to_numpy(float), nan=1.0)
        raw = np.nan_to_num(frame["raw_residual"].to_numpy(float), nan=np.inf)
        return np.lexsort((ids, raw, score))
    if method == "domain":
        score = np.nan_to_num(frame["risk_domain"].to_numpy(float), nan=1.0)
        invalid = np.nan_to_num(frame["raw_invalid"].to_numpy(float), nan=1.0)
        distance = np.nan_to_num(
            frame["raw_domain_distance"].to_numpy(float), nan=-np.inf
        )
        return np.lexsort((ids, -distance, invalid, score))
    raise ValueError(f"Unknown internal method: {method}")


def internal_review_mask(
    frame: pd.DataFrame, method: str, fraction: float = REVIEW_FRACTION
) -> pd.Series:
    selected = pd.Series(False, index=frame.index)
    for _, part in frame.groupby("k", sort=True):
        quota = max(1, int(round(fraction * len(part))))
        order = internal_low_to_high_order(part, method)
        selected.loc[part.iloc[order[-quota:]].index] = True
    return selected
