"""
Deterministic synthetic hyperspectral cube for self-contained tests.

Produces a [Z, Y, X] unsigned-integer cube with realistic structure:
  * a handful of smooth spectral "endmembers" (so adjacent bands are highly
    correlated -> exercises the spectral predictor), and
  * smooth spatial abundance maps (so adjacent pixels are correlated -> exercises
    the spatial predictor), plus mild noise (an incompressible component).

It is pure numpy (no scipy / no data files) so the test suite runs from a fresh
clone, and it compresses at a realistic lossless ratio (~2-3:1), similar to real
scenes like Indian Pines.
"""

from __future__ import annotations

import numpy as np


def _smooth(a: np.ndarray, iters: int) -> np.ndarray:
    """Separable 5-point box blur via np.roll (toroidal, pure numpy)."""
    for _ in range(iters):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
             + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    return a


def make_synthetic_hsi(num_bands: int = 50, height: int = 64, width: int = 64,
                       num_materials: int = 6, dynamic_range: int = 16,
                       noise: float = 6.0, fill_fraction: float = 0.15,
                       seed: int = 0) -> np.ndarray:
    """Return a deterministic [Z, Y, X] int64 hyperspectral cube.

    Args:
        num_bands/height/width: cube dimensions (Z, Y, X).
        num_materials: number of distinct smooth spectral signatures mixed in.
        dynamic_range: bit depth D; samples lie in [0, 2^D - 1].
        noise: std-dev (in counts) of additive white noise.
        fill_fraction: peak signal as a fraction of 2^D (real scenes rarely fill
            the full range; ~0.15 mimics Indian Pines' ~9600/65535).
        seed: RNG seed (fully deterministic output).
    """
    rng = np.random.RandomState(seed)
    Z, Y, X = num_bands, height, width

    # Smooth spectral endmembers: each a sum of a few Gaussian bumps over bands.
    band_axis = np.linspace(0.0, 1.0, Z)
    spectra = np.empty((num_materials, Z))
    for k in range(num_materials):
        s = np.full(Z, rng.uniform(0.1, 0.4))            # baseline reflectance
        for _ in range(rng.randint(2, 5)):
            c, w, a = rng.uniform(0, 1), rng.uniform(0.05, 0.25), rng.uniform(0.3, 1.0)
            s += a * np.exp(-((band_axis - c) ** 2) / (2 * w * w))
        spectra[k] = s

    # Smooth spatial abundance maps, softmax-normalised across materials.
    abund = np.stack([_smooth(rng.rand(Y, X), iters=6) for _ in range(num_materials)])
    abund = np.exp(3.0 * abund)
    abund /= abund.sum(axis=0, keepdims=True)

    cube = np.einsum("kyx,kz->zyx", abund, spectra)      # [Z, Y, X] mixed reflectance
    cube -= cube.min()
    peak = (1 << dynamic_range) - 1
    cube = cube / cube.max() * (peak * fill_fraction)
    cube += rng.randn(Z, Y, X) * noise                   # mild sensor noise
    return np.clip(np.round(cube), 0, peak).astype(np.int64)


if __name__ == "__main__":
    c = make_synthetic_hsi(40, 48, 48)
    print(f"shape={c.shape} dtype={c.dtype} min={c.min()} max={c.max()} "
          f"mean={c.mean():.1f}")
