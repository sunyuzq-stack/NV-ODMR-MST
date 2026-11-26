"""Selected spectral diagnostics and their source dependencies."""
from types import SimpleNamespace

import numpy as np

from odmr_core import _SQRT_2LN2


def recon_batch(fg: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.atleast_2d(np.asarray(p, float))
    fm = (fg * 1e3)[None, :]
    c = p[:, 0:1] * 1e3
    sp, w1, w2 = p[:, 1:2], p[:, 2:3], p[:, 3:4]
    c1, c2, eta = p[:, 4:5], p[:, 5:6], p[:, 6:7]
    o, sl, qd = p[:, 7:8], p[:, 8:9], p[:, 9:10]
    x1, x2 = c - sp / 2.0, c + sp / 2.0
    t = ((fg - fg[0]) / (fg[-1] - fg[0]))[None, :]
    bl = o + sl * (t - 0.5) + qd * (t - 0.5) ** 2
    s1, s2 = w1 / _SQRT_2LN2, w2 / _SQRT_2LN2
    L1 = 1.0 / (1.0 + ((fm - x1) / w1) ** 2)
    G1 = np.exp(-0.5 * ((fm - x1) / s1) ** 2)
    L2 = 1.0 / (1.0 + ((fm - x2) / w2) ** 2)
    G2 = np.exp(-0.5 * ((fm - x2) / s2) ** 2)
    m1 = eta * L1 + (1 - eta) * G1
    m2 = eta * L2 + (1 - eta) * G2
    return bl * (1.0 - c1 * m1 - c2 * m2)

# Bind the original residual function to its two cvf dependencies.
cvf = SimpleNamespace(NPAR=10, recon_batch=recon_batch)


class AuditFeatureError(RuntimeError):
    """Input arrays do not follow the original audit contract."""


def weighted_affine_residual(
    frequency_ghz: np.ndarray,
    observed_counts: np.ndarray,
    sigma_mean_counts: np.ndarray,
    parameters: np.ndarray,
) -> dict[str, np.ndarray]:
    """Noise-standardized residual after removing count scale and offset.

    The pseudo-Voigt shape is in relative-intensity units whereas the camera
    spectra are in counts.  Weighted least squares projects out only the two
    nuisance directions ``scale`` and ``offset``.  The returned score is the
    square root of chi-square divided by (number of valid frequencies - 2).
    It is therefore a dimensionless audit score, not a claim that the CNN
    performed a 10-parameter WLS fit to this spectrum.
    """
    y = np.asarray(observed_counts, dtype=np.float64)
    sigma = np.asarray(sigma_mean_counts, dtype=np.float64)
    p = np.asarray(parameters, dtype=np.float64)
    if y.ndim != 2 or sigma.shape != y.shape or p.shape != (y.shape[0], cvf.NPAR):
        raise AuditFeatureError(
            f"Residual shape mismatch: y={y.shape}, sigma={sigma.shape}, p={p.shape}"
        )

    model = cvf.recon_batch(np.asarray(frequency_ghz, dtype=np.float64), p)
    valid = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(model)
    weight = np.where(valid, 1.0 / np.maximum(sigma, 1e-12) ** 2, 0.0)
    yy = np.where(valid, y, 0.0)
    mm = np.where(valid, model, 0.0)

    sw = weight.sum(axis=1)
    swm = (weight * mm).sum(axis=1)
    smm = (weight * mm * mm).sum(axis=1)
    swy = (weight * yy).sum(axis=1)
    smy = (weight * mm * yy).sum(axis=1)
    determinant = smm * sw - swm**2
    solvable = np.abs(determinant) > 1e-30
    scale = np.divide(
        smy * sw - swm * swy,
        determinant,
        out=np.full(y.shape[0], np.nan),
        where=solvable,
    )
    offset = np.divide(
        smm * swy - swm * smy,
        determinant,
        out=np.full(y.shape[0], np.nan),
        where=solvable,
    )
    fitted = scale[:, None] * model + offset[:, None]
    z_residual = np.where(valid, (y - fitted) / sigma, np.nan)
    n_valid = valid.sum(axis=1)
    dof = n_valid - 2
    chi2 = np.nansum(z_residual**2, axis=1)
    chi2_reduced = np.divide(
        chi2,
        dof,
        out=np.full(y.shape[0], np.nan),
        where=solvable & (dof > 0),
    )
    score = np.sqrt(chi2_reduced)

    # A scale-free shape RMS is saved as a diagnostic, not used by the audit.
    y_mean = np.nanmean(np.where(valid, y, np.nan), axis=1, keepdims=True)
    y_std = np.nanstd(np.where(valid, y, np.nan), axis=1, keepdims=True)
    m_mean = np.nanmean(np.where(valid, model, np.nan), axis=1, keepdims=True)
    m_std = np.nanstd(np.where(valid, model, np.nan), axis=1, keepdims=True)
    yz = (y - y_mean) / np.maximum(y_std, 1e-12)
    mz = (model - m_mean) / np.maximum(m_std, 1e-12)
    shape_rms = np.sqrt(np.nanmean(np.where(valid, (yz - mz) ** 2, np.nan), axis=1))

    return {
        "score": score,
        "chi2_reduced": chi2_reduced,
        "affine_scale": scale,
        "affine_offset": offset,
        "n_valid": n_valid.astype(np.int32),
        "shape_rms_z": shape_rms,
    }


def doublet_resolvability(parameters):
    """Return splitting divided by the mean half width at half maximum."""
    width = 0.5 * (parameters[:, 2] + parameters[:, 3])
    return np.divide(parameters[:, 1], width, out=np.full(width.shape, np.nan), where=width > 0)
