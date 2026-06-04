"""
Utility functions: image I/O, noise addition, quality metrics.
"""

import numpy as np
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Union[str, Path], as_float: bool = True) -> np.ndarray:
    """Load a grayscale image. Returns float64 in [0, 255] by default."""
    from PIL import Image
    img = np.array(Image.open(path).convert('L'), dtype=np.float64)
    return img


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """Save a float image (assumed [0, 255] range) as PNG."""
    from PIL import Image
    arr = np.clip(np.round(image), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: int = 42,
) -> np.ndarray:
    """Add AWGN with standard deviation `sigma` to the image."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image.shape)
    return image.astype(float) + noise


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((reference.astype(float) - estimate.astype(float)) ** 2))


def psnr(reference: np.ndarray, estimate: np.ndarray, peak: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    err = mse(reference, estimate)
    if err == 0.0:
        return float('inf')
    return float(10.0 * np.log10(peak ** 2 / err))


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Structural Similarity Index (Wang et al., 2004)."""
    from skimage.metrics import structural_similarity
    r = reference.astype(float)
    e = estimate.astype(float)
    data_range = r.max() - r.min()
    return float(structural_similarity(r, e, data_range=data_range))


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def show_results(
    clean: np.ndarray,
    noisy: np.ndarray,
    *denoised_pairs,            # (label, image) pairs
    cmap: str = 'gray',
    save_path: Union[str, Path, None] = None,
) -> None:
    """
    Display clean / noisy / denoised images side by side with PSNR labels.

    Args:
        clean            : ground-truth image
        noisy            : noisy image
        denoised_pairs   : any number of (label, denoised_image) tuples
        cmap             : matplotlib colormap
        save_path        : if given, save figure instead of showing
    """
    import matplotlib.pyplot as plt

    n_cols = 2 + len(denoised_pairs)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    def _show(ax, img, title):
        ax.imshow(np.clip(img, 0, 255), cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    _show(axes[0], clean, 'Clean')
    noisy_psnr = psnr(clean, noisy)
    _show(axes[1], noisy, f'Noisy\nPSNR={noisy_psnr:.2f} dB')

    for ax, (label, img) in zip(axes[2:], denoised_pairs):
        p = psnr(clean, img)
        _show(ax, img, f'{label}\nPSNR={p:.2f} dB')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
