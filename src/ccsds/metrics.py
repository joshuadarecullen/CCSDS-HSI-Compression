"""
Quality metrics for (near-lossless) compression: PSNR, MSSIM, SAM.

numpy-only. Images are [Z, Y, X] (bands, height, width), matching the codec.
For lossless output PSNR is +inf, MSSIM is 1 and SAM is 0; these are useful for
comparing near-lossless reconstructions.
"""

from __future__ import annotations

import numpy as np


def _as_float(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def calculate_psnr(original, reconstructed, dynamic_range: int, peak=None) -> float:
    """Peak signal-to-noise ratio in dB. `peak` defaults to 2^dynamic_range - 1 (the
    full D-bit range); pass the data's actual maximum for a tighter reference.

    Returns +inf for an exact (lossless) match.
    """
    x, y = _as_float(original), _as_float(reconstructed)
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    if peak is None:
        peak = (1 << dynamic_range) - 1
    return float(20 * np.log10(peak) - 10 * np.log10(mse))


def _gaussian1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable Gaussian blur over a 2D band, 'valid' borders."""
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, img)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, out)


def _ssim_band(x: np.ndarray, y: np.ndarray, data_range: float,
               sigma: float = 1.5, win: int = 11) -> float:
    """SSIM of one band (Wang et al.), Gaussian-windowed and averaged over the map."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    win = min(win, x.shape[0], x.shape[1])
    if win < 3:                                   # too small to window: fall back to global stats
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        vxy = np.mean((x - mx) * (y - my))
        return ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    if win % 2 == 0:
        win -= 1
    k = _gaussian1d(sigma, win // 2)
    mx, my = _blur_valid(x, k), _blur_valid(y, k)
    vx = _blur_valid(x * x, k) - mx * mx
    vy = _blur_valid(y * y, k) - my * my
    vxy = _blur_valid(x * y, k) - mx * my
    ssim_map = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(ssim_map.mean())


def calculate_mssim(original, reconstructed, data_range=None) -> float:
    """Mean structural similarity, averaged over bands; in [-1, 1], 1 = identical.

    `data_range` defaults to the original's value range (max - min).
    """
    x, y = _as_float(original), _as_float(reconstructed)
    if data_range is None:
        data_range = float(x.max() - x.min()) or 1.0
    return float(np.mean([_ssim_band(x[z], y[z], data_range) for z in range(x.shape[0])]))


def calculate_spectral_angle(original, reconstructed) -> float:
    """Mean spectral angle mapper (SAM) over all pixels, in radians; 0 = identical spectra."""
    x, y = _as_float(original), _as_float(reconstructed)
    dot = np.sum(x * y, axis=0)
    denom = np.sqrt(np.sum(x * x, axis=0)) * np.sqrt(np.sum(y * y, axis=0))
    cos = np.ones_like(dot)
    good = denom > 1e-12
    cos[good] = np.clip(dot[good] / denom[good], -1.0, 1.0)
    return float(np.mean(np.arccos(cos)))


def quality_report(original, reconstructed, dynamic_range: int, peak=None) -> dict:
    """PSNR (dB), MSSIM, SAM (rad) and max absolute error, as a dict."""
    x, y = _as_float(original), _as_float(reconstructed)
    return {
        "psnr_db": calculate_psnr(x, y, dynamic_range, peak),
        "mssim": calculate_mssim(x, y),
        "sam_rad": calculate_spectral_angle(x, y),
        "max_abs_error": float(np.abs(x - y).max()),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.integers(0, 1 << 16, size=(8, 32, 32)).astype(np.int64)
    assert calculate_psnr(img, img, 16) == float("inf")
    assert calculate_mssim(img, img) > 0.999999
    assert calculate_spectral_angle(img, img) < 1e-5      # ~0; arccos is touchy near 1
    noisy = np.clip(img + rng.integers(-3, 4, img.shape), 0, (1 << 16) - 1)
    rep = quality_report(img, noisy, 16)
    assert np.isfinite(rep["psnr_db"]) and rep["mssim"] < 1.0 and rep["sam_rad"] > 0
    print(f"identical -> inf dB, MSSIM 1, SAM 0; noisy -> {rep}")
    print("metrics self-test OK")
