"""
Demo: DFB-based image denoising vs DWT baseline.

Reproduces the comparison from Rosiles & Smith (ICIP 2000) and evaluates
the proposed improvements (BayesShrink, multi-scale DFB).

Usage:
    python demo.py [--image PATH] [--sigma 30] [--output results/]
"""

import argparse
import numpy as np
from pathlib import Path

from src.utils import load_image, add_gaussian_noise, psnr, mse, show_results, save_image
from src.denoising import dfb_denoise, multiscale_dfb_denoise


def dwt_denoise(noisy: np.ndarray, sigma: float, wavelet: str = 'db4', level: int = 4) -> np.ndarray:
    """Wavelet soft-thresholding baseline (BayesShrink per subband)."""
    import pywt
    coeffs = pywt.wavedec2(noisy, wavelet, level=level)
    new_coeffs = [coeffs[0]]  # keep approximation
    for detail_tuple in coeffs[1:]:
        new_detail = []
        for subband in detail_tuple:
            sigma_y = float(np.std(subband))
            sigma_x = np.sqrt(max(sigma_y**2 - sigma**2, 1e-10))
            t = sigma**2 / sigma_x
            new_detail.append(np.sign(subband) * np.maximum(np.abs(subband) - t, 0))
        new_coeffs.append(tuple(new_detail))
    return pywt.waverec2(new_coeffs, wavelet)


def run_demo(image_path: str, sigma: float, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clean = load_image(image_path)
    noisy = add_gaussian_noise(clean, sigma)

    print(f"Image : {image_path}  shape={clean.shape}")
    print(f"Sigma : {sigma:.1f}")
    print(f"Noisy PSNR: {psnr(clean, noisy):.2f} dB   MSE: {mse(clean, noisy):.2f}\n")

    results = []

    # --- Paper methods (SURE, GCV, MAD) with undecimated DFB ---
    for method in ('sure', 'gcv', 'mad'):
        den = dfb_denoise(noisy, sigma=sigma, n_bands=8, method=method)
        p = psnr(clean, den)
        m = mse(clean, den)
        label = f'DFB-{method.upper()}'
        print(f"{label:<20} PSNR={p:.2f} dB   MSE={m:.2f}")
        results.append((label, den))
        save_image(den, out / f'{label}.png')

    # --- Improvement: BayesShrink DFB ---
    den_bayes = dfb_denoise(noisy, sigma=sigma, n_bands=8, method='bayes')
    p = psnr(clean, den_bayes)
    m = mse(clean, den_bayes)
    print(f"{'DFB-Bayes':<20} PSNR={p:.2f} dB   MSE={m:.2f}")
    results.append(('DFB-Bayes', den_bayes))
    save_image(den_bayes, out / 'DFB-Bayes.png')

    # --- Improvement: 16-band DFB ---
    den_16 = dfb_denoise(noisy, sigma=sigma, n_bands=16, method='bayes')
    p = psnr(clean, den_16)
    m = mse(clean, den_16)
    print(f"{'DFB-16bands':<20} PSNR={p:.2f} dB   MSE={m:.2f}")
    results.append(('DFB-16bands', den_16))
    save_image(den_16, out / 'DFB-16bands.png')

    # --- Improvement: Multi-scale DFB ---
    den_ms = multiscale_dfb_denoise(noisy, sigma=sigma, n_scales=3, n_bands=8, method='sure')
    p = psnr(clean, den_ms)
    m = mse(clean, den_ms)
    print(f"{'Multi-scale DFB':<20} PSNR={p:.2f} dB   MSE={m:.2f}")
    results.append(('MS-DFB', den_ms))
    save_image(den_ms, out / 'MS-DFB.png')

    # --- DWT baseline ---
    try:
        den_dwt = dwt_denoise(noisy, sigma)
        p = psnr(clean, den_dwt)
        m = mse(clean, den_dwt)
        print(f"{'DWT-Bayes':<20} PSNR={p:.2f} dB   MSE={m:.2f}")
        results.append(('DWT-Bayes', den_dwt))
        save_image(den_dwt, out / 'DWT-Bayes.png')
    except ImportError:
        print("pywt not available — skipping DWT baseline")

    # Save visual comparison
    show_results(
        clean, noisy,
        *results[:4],          # show first 4 denoised for readability
        save_path=out / 'comparison.png'
    )
    print(f"\nResults saved to {out}/")


def _build_test_image() -> str:
    """Use scikit-image's camera image as the default test image."""
    from skimage import data
    from src.utils import save_image as _save

    img = data.camera().astype(np.float64)   # 512×512 grayscale
    path = '/tmp/test_image.png'
    _save(img, path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DFB image denoising demo')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input grayscale image (PNG/JPEG). '
                             'Uses a synthetic image if not provided.')
    parser.add_argument('--sigma', type=float, default=30.0,
                        help='Gaussian noise std (default: 30)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results/)')
    args = parser.parse_args()

    image_path = args.image or _build_test_image()
    run_demo(image_path, args.sigma, args.output)
