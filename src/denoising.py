"""
DFB-based image denoising pipelines.

Paper pipeline (dfb_denoise):
    image → Gaussian LP/HP split → DFB on HP → threshold → LP + denoised HP

Multi-scale improvement (multiscale_dfb_denoise):
    image → undecimated Gaussian pyramid → DFB per scale → threshold → sum

Supported threshold methods: 'sure', 'gcv', 'mad', 'bayes'
Supported threshold operators: 'soft', 'hard'
"""

import numpy as np
from typing import List, Optional

from .dfb import dfb_decompose, dfb_reconstruct
from .threshold import (
    estimate_sigma,
    soft_threshold,
    hard_threshold,
    sure_threshold,
    gcv_threshold,
    bayes_threshold,
    universal_threshold,
)


_THRESHOLD_FNS = {
    'sure' : sure_threshold,
    'gcv'  : gcv_threshold,
    'bayes': bayes_threshold,
    'mad'  : universal_threshold,   # MAD noise est. + universal threshold
}

_OPERATOR_FNS = {
    'soft': soft_threshold,
    'hard': hard_threshold,
}


def _denoise_subbands(
    subbands: List[np.ndarray],
    sigma_global: float,
    method: str,
    operator: str,
) -> List[np.ndarray]:
    """
    Apply threshold estimation + operator to each subband.

    For undecimated DFB with partition-of-unity filters the noise power in each
    subband is approximately σ²/n_bands (energy equally distributed).  Adaptive
    methods (SURE, GCV) infer the threshold purely from the data; parametric
    methods (bayes, mad) require the per-subband noise sigma.
    """
    threshold_fn = _THRESHOLD_FNS[method]
    operator_fn  = _OPERATOR_FNS[operator]
    n_bands = len(subbands)

    # Per-subband noise sigma: noise energy divides equally over n_bands
    sigma_k = sigma_global / np.sqrt(n_bands)

    denoised = []
    for band in subbands:
        t = threshold_fn(band, sigma_k)
        denoised.append(operator_fn(band, t))
    return denoised


def _estimate_sigma_from_image(img: np.ndarray) -> float:
    """
    Estimate noise sigma from the finest-scale wavelet diagonal subband (MAD).
    Standard practice from Donoho & Johnstone (1995).
    Falls back to MAD of the raw image if pywt is unavailable.
    """
    try:
        import pywt
        _, (_, _, d) = pywt.dwt2(img, 'db1')
        return float(np.median(np.abs(d)) / 0.6745)
    except ImportError:
        return float(np.median(np.abs(img - np.median(img))) / 0.6745)


def dfb_denoise(
    noisy: np.ndarray,
    sigma: Optional[float] = None,
    n_bands: int = 8,
    method: str = 'sure',
    operator: str = 'soft',
    lp_sigma: float = 2.0,
) -> np.ndarray:
    """
    DFB-based image denoiser (reimplementation of Rosiles & Smith, ICIP 2000).

    Key insight for the undecimated implementation: directional wedge filters
    cover the full spectrum, so subbands are dense (not sparse).  We first
    separate a smooth LP component (kept intact) and apply the DFB only to
    the HP residual, which IS sparse at directional edges — making thresholding
    effective.  This is equivalent to the critically-sampled DFB in the paper
    but avoids aliasing artefacts.

    Args:
        noisy    : noisy grayscale image (float, any range)
        sigma    : noise std; estimated from finest wavelet band if None
        n_bands  : directional subbands (default 8, as in paper)
        method   : 'sure' | 'gcv' | 'mad' | 'bayes'
        operator : 'soft' | 'hard'
        lp_sigma : spatial sigma for the Gaussian LP/HP split (default 2.0)

    Returns:
        Denoised image.
    """
    from scipy.ndimage import gaussian_filter

    if method not in _THRESHOLD_FNS:
        raise ValueError(f"method must be one of {list(_THRESHOLD_FNS)}")
    if operator not in _OPERATOR_FNS:
        raise ValueError(f"operator must be one of {list(_OPERATOR_FNS)}")

    img = noisy.astype(float)

    # LP/HP separation: LP is approximately noise-free (Gaussian blur averages noise)
    lp = gaussian_filter(img, sigma=lp_sigma)
    hp = img - lp   # HP = directional edges/textures + noise (SPARSE signal)

    if sigma is None:
        sigma = _estimate_sigma_from_image(img)

    # Apply DFB to the sparse HP residual
    subbands, _ = dfb_decompose(hp, n_bands)
    subbands_d  = _denoise_subbands(subbands, sigma, method, operator)
    hp_denoised = dfb_reconstruct(subbands_d)

    return lp + hp_denoised


def multiscale_dfb_denoise(
    noisy: np.ndarray,
    sigma: Optional[float] = None,
    n_scales: int = 3,
    n_bands: int = 8,
    method: str = 'sure',
    operator: str = 'soft',
) -> np.ndarray:
    """
    Multi-scale undecimated DFB denoiser.  [IMPROVEMENT]

    Decomposes the image into n_scales bandpass layers using an undecimated
    Gaussian pyramid (no decimation → perfect reconstruction from unmodified
    subbands).  DFB is applied to each bandpass layer to get directional
    subbands at that scale, which are then thresholded independently.

    This achieves simultaneous scale and direction selectivity, related to the
    Contourlet transform (Do & Vetterli, 2005), without any reconstruction
    artefacts from decimation.

    Args:
        noisy   : noisy grayscale image
        sigma   : global noise std; estimated from image if None
        n_scales: number of bandpass layers (default 3)
        n_bands : directional subbands per scale (default 8)
        method  : 'sure' | 'gcv' | 'mad' | 'bayes'
        operator: 'soft' | 'hard'

    Returns:
        Denoised image.
    """
    from scipy.ndimage import gaussian_filter

    img = noisy.astype(float)
    if sigma is None:
        sigma = _estimate_sigma_from_image(img)

    # Build undecimated bandpass layers:
    # layer[l] = LP[l-1] - LP[l], where LP[l] = gaussian_filter(img, 2^l)
    lp_prev = img
    layers: List[np.ndarray] = []
    for l in range(n_scales):
        lp_curr = gaussian_filter(img, sigma=float(2 ** (l + 1)))
        layers.append(lp_prev - lp_curr)
        lp_prev = lp_curr
    coarsest_lp = lp_prev   # coarsest smooth residual (kept intact)

    # Denoise each bandpass layer with DFB + threshold
    denoised = coarsest_lp.copy()
    for layer in layers:
        layer_sigma = _estimate_sigma_from_image(layer)
        subbands, _ = dfb_decompose(layer, n_bands)
        subbands_d  = _denoise_subbands(subbands, layer_sigma, method, operator)
        denoised += dfb_reconstruct(subbands_d)

    return denoised
