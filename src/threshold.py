"""
Threshold estimation methods for image denoising.

Implements the methods from Rosiles & Smith (ICIP 2000):
  - MAD  : noise estimation via Median Absolute Deviation
  - SURE : Stein's Unbiased Risk Estimate  (Donoho & Johnstone 1995)
  - GCV  : Generalised Cross Validation    (Craven & Wahba 1979)

Plus improvements:
  - BayesShrink : per-subband Bayesian threshold (Chang et al. 2000)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------

def estimate_sigma(subband: np.ndarray) -> float:
    """
    Estimate noise std from a subband using the MAD estimator.
    Robust because signal energy tends to produce a few large coefficients,
    while noise gives many small ones near zero.

    σ̂ = median(|y|) / 0.6745
    """
    return float(np.median(np.abs(subband))) / 0.6745


# ---------------------------------------------------------------------------
# Thresholding operators
# ---------------------------------------------------------------------------

def soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    """Soft (shrinkage) thresholding: sign(x) * max(|x| - t, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def hard_threshold(x: np.ndarray, t: float) -> np.ndarray:
    """Hard thresholding: keep x if |x| > t, else 0."""
    return x * (np.abs(x) > t)


# ---------------------------------------------------------------------------
# Threshold selection rules
# ---------------------------------------------------------------------------

def sure_threshold(y: np.ndarray, sigma: float) -> float:
    """
    SURE (Stein's Unbiased Risk Estimate) threshold for soft thresholding.

    Works on normalized coefficients z = y/σ where the model is z = x/σ + ε,
    ε ~ N(0,1).  The SURE risk estimate for soft threshold t̃ (in z-domain) is:

        SURE(t̃) = n - 2·#{|z_i| ≤ t̃} + Σ_i min(z_i², t̃²)

    The returned threshold is t* = t̃* × σ (back-scaled to original units).

    Args:
        y    : noisy subband coefficients
        sigma: noise standard deviation for this subband

    Returns:
        Optimal soft-threshold value t* ≥ 0 (in same units as y).
    """
    y_flat = y.ravel().astype(float)
    n = y_flat.size
    if sigma < 1e-12:
        return 0.0

    # Normalise: work in units of sigma so the SURE formula assumes N(0,1) noise
    z = y_flat / sigma
    abs_z = np.sort(np.abs(z))

    cum_z2 = np.cumsum(abs_z ** 2)
    k      = np.arange(n)
    below  = k + 1
    above  = n - below
    sum_min = cum_z2 + abs_z ** 2 * above     # Σ min(z_i², t̃²)

    sure_vals = n - 2.0 * below + sum_min
    best_k    = int(np.argmin(sure_vals))
    return float(abs_z[best_k] * sigma)       # back-scale to original units


def gcv_threshold(y: np.ndarray, sigma: float) -> float:
    """
    GCV (Generalised Cross Validation) threshold for soft thresholding.

    Normalises by σ so the criterion is scale-independent, then back-scales.

        GCV(t̃) = Σ_i min(z_i², t̃²)  /  (#{|z_i| ≤ t̃} / n)²

    where z_i = y_i / σ.

    Args:
        y    : noisy subband coefficients
        sigma: noise std for this subband

    Returns:
        Optimal threshold t* (same units as y).
    """
    y_flat = y.ravel().astype(float)
    n = y_flat.size
    if sigma < 1e-12:
        return 0.0

    z      = y_flat / sigma
    abs_z  = np.sort(np.abs(z))
    cum_z2 = np.cumsum(abs_z ** 2)

    k      = np.arange(n)
    below  = k + 1                             # #{|z_i| ≤ t̃}
    above  = n - below
    sum_min = cum_z2 + abs_z ** 2 * above

    # Denominator: (below/n)² — avoids divide-by-zero for the smallest candidate
    denom    = (below / n) ** 2
    denom    = np.where(denom < 1e-12, 1e-12, denom)
    gcv_vals = sum_min / denom

    best_k = int(np.argmin(gcv_vals))
    return float(abs_z[best_k] * sigma)


def bayes_threshold(y: np.ndarray, sigma: float) -> float:
    """
    BayesShrink threshold (Chang, Yu & Vetterli, 2000).  [IMPROVEMENT]

    Assumes the signal has a generalised Gaussian prior.  The threshold is:
        t* = σ² / σ_x
    where σ_x = sqrt(max(σ_y² - σ², 0)) is the estimated signal std.

    Tends to give subband-adaptive thresholds better tuned to local content
    than the global SURE/GCV estimators.

    Args:
        y    : noisy subband coefficients
        sigma: noise std

    Returns:
        Threshold t* ≥ 0.
    """
    sigma2 = sigma ** 2
    sigma_y2 = float(np.var(y))
    sigma_x = np.sqrt(max(sigma_y2 - sigma2, 0.0))
    if sigma_x < 1e-10:
        # No signal detected → zero everything out (pure noise subband)
        return float(np.max(np.abs(y)))
    return float(sigma2 / sigma_x)


def universal_threshold(y: np.ndarray, sigma: float) -> float:
    """
    Donoho-Johnstone universal threshold: σ * sqrt(2 * log(n)).
    Simple baseline used in conjunction with MAD noise estimation.
    """
    n = y.size
    return float(sigma * np.sqrt(2.0 * np.log(n)))
