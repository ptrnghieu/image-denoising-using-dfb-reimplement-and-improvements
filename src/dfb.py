"""
Directional Filter Bank (DFB) — undecimated frequency-domain implementation.

The DFB partitions the 2D frequency plane into N directional wedge-shaped subbands.
We use an undecimated (shift-invariant) version for better denoising quality compared
to the critically-sampled DFB in the original paper (Rosiles & Smith, ICIP 2000).

The frequency partition uses raised-cosine windowed wedges that sum to 1 (partition
of unity), guaranteeing perfect reconstruction from unmodified subbands.
"""

import numpy as np
from typing import List, Tuple


def _angle_grid(shape: Tuple[int, int]) -> np.ndarray:
    """Angle (in radians) at each point in the 2D DFT frequency grid."""
    M, N = shape
    # fftfreq returns frequencies in [-0.5, 0.5)
    # We use arctan2(row_freq, col_freq) to get angle in [-pi, pi]
    w_row = np.fft.fftfreq(M)
    w_col = np.fft.fftfreq(N)
    W_col, W_row = np.meshgrid(w_col, w_row)
    return np.arctan2(W_row, W_col)


def _smooth_wedge(theta: np.ndarray, center: float, half_bw: float) -> np.ndarray:
    """
    Raised-cosine wedge window centered at `center` (rad) with half-bandwidth `half_bw`.
    Returns values in [0, 1].
    """
    diff = theta - center
    # Wrap to [-pi, pi]
    diff = (diff + np.pi) % (2.0 * np.pi) - np.pi

    H = np.zeros_like(theta)
    inner = np.abs(diff) <= half_bw * 0.5
    H[inner] = 1.0
    outer = (~inner) & (np.abs(diff) <= half_bw)
    H[outer] = 0.5 * (1.0 + np.cos(np.pi * (np.abs(diff[outer]) - half_bw * 0.5) / (half_bw * 0.5)))
    return H


def create_dfb_filters(n_bands: int, shape: Tuple[int, int]) -> List[np.ndarray]:
    """
    Build N frequency-domain directional filters that partition the 2D spectrum.

    For a real image the DFT satisfies conjugate symmetry, so directions θ and
    θ+π are equivalent.  We therefore span [0, π) and replicate to [-π, 0).

    Args:
        n_bands : number of directional subbands (power of 2 recommended, e.g. 8)
        shape   : (rows, cols) of the image to be filtered

    Returns:
        List of n_bands 2D arrays (same shape as image), each summing to ≤ 1
        and collectively summing to 1 everywhere (partition of unity).
    """
    theta = _angle_grid(shape)

    delta = np.pi / n_bands          # half-bandwidth of one band over [0, π)
    overlap = 1.25                    # slight overlap for smooth partition

    filters: List[np.ndarray] = []
    for k in range(n_bands):
        # Center angle in [−π/2, π/2) so bands tile the half-plane
        center = k * delta - np.pi / 2.0 + delta / 2.0
        # Build wedge and its π-rotated copy (conjugate symmetry of real images)
        H = _smooth_wedge(theta, center, delta * overlap)
        H += _smooth_wedge(theta, center + np.pi, delta * overlap)
        filters.append(H)

    # Normalise so the filters form a partition of unity
    total = sum(filters)
    total = np.where(total < 1e-12, 1.0, total)
    filters = [H / total for H in filters]
    return filters


def dfb_decompose(
    image: np.ndarray,
    n_bands: int = 8,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Undecimated DFB decomposition.

    Each subband y_k = IFFT( FFT(image) * H_k ) where H_k is the k-th
    directional filter.  Because sum_k H_k = 1, summing the subbands
    reconstructs the original image exactly.

    Args:
        image  : 2-D grayscale float array
        n_bands: number of directional subbands (default 8, as in the paper)

    Returns:
        subbands : list of n_bands real-valued subband images (same size as input)
        filters  : list of n_bands frequency-domain masks (for reconstruction)
    """
    X = np.fft.fft2(image.astype(float))
    filters = create_dfb_filters(n_bands, image.shape)

    subbands = [np.real(np.fft.ifft2(X * H)) for H in filters]
    return subbands, filters


def dfb_reconstruct(subbands: List[np.ndarray]) -> np.ndarray:
    """
    Reconstruct image from (possibly modified) DFB subbands.

    For unmodified subbands this is perfect reconstruction.
    After soft/hard thresholding it gives the denoised image.

    Args:
        subbands: list of directional subband images

    Returns:
        Reconstructed (or denoised) image as a 2-D float array.
    """
    return sum(subbands)


# ---------------------------------------------------------------------------
# Laplacian Pyramid helpers (used by the multi-scale improvement)
# ---------------------------------------------------------------------------

def _gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Simple isotropic Gaussian blur via separable 1-D convolutions."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image.astype(float), sigma=sigma)


_LP_SIGMA = 1.2   # Gaussian sigma for LP pyramid; shared by decompose & reconstruct


def lp_decompose(image: np.ndarray, n_levels: int = 3) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Laplacian Pyramid decomposition (Burt & Adelson, 1983).

    Returns:
        residuals : list of n_levels band-pass (Laplacian) images, finest first
        lowpass   : coarsest lowpass residual
    """
    residuals: List[np.ndarray] = []
    current = image.astype(float)
    for _ in range(n_levels):
        low = _gaussian_blur(current, sigma=_LP_SIGMA)
        residuals.append(current - low)
        current = low[::2, ::2]          # decimate by 2
    return residuals, current


def _upsample2x(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Upsample by 2 using the same Gaussian kernel as lp_decompose.
    Insert zeros at odd-indexed positions then apply Gaussian × 4 to fill them.
    """
    M, N = target_shape
    up = np.zeros((M, N))
    up[::2, ::2] = image
    return _gaussian_blur(up, sigma=_LP_SIGMA) * 4.0


def lp_reconstruct(residuals: List[np.ndarray], lowpass: np.ndarray) -> np.ndarray:
    """
    Reconstruct image from Laplacian Pyramid.
    Uses the conjugate Gaussian filter (× 4 after zero-insertion) matching
    lp_decompose so that decompose → reconstruct is near-lossless.
    """
    current = lowpass.astype(float)
    for res in reversed(residuals):
        up = _upsample2x(current, res.shape)
        current = up + res
    return current
