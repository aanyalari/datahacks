"""Cross-wavelet style coherence between two uniformly sampled series (Morlet CWT)."""

from __future__ import annotations

import numpy as np

try:
    import pywt

    _HAS_PYWT = True
except ImportError:
    pywt = None
    _HAS_PYWT = False


def morlet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    sampling_period: float = 1.0,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Return (periods, coherence, scales) where coherence ~ |Wxy|^2 / (|Wx|^2 |Wy|^2) per scale.
    Series are standardized; NaNs dropped pairwise.
    """
    if not _HAS_PYWT:
        return None, None, None
    m = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[m], dtype=float)
    y = np.asarray(y[m], dtype=float)
    if len(x) < 64 or len(y) < 64:
        return None, None, None
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    if scales is None:
        scales = np.logspace(0.3, 2.2, num=48)
    Wx, _ = pywt.cwt(x, scales, "morl", sampling_period=sampling_period)
    Wy, _ = pywt.cwt(y, scales, "morl", sampling_period=sampling_period)
    wxy = Wx * np.conj(Wy)
    num = np.abs(wxy) ** 2
    den = (np.abs(Wx) ** 2) * (np.abs(Wy) ** 2) + 1e-12
    coh = (num / den).mean(axis=1)
    coh = np.clip(np.real(coh), 0.0, 1.0)
    # Morlet Fourier wavelength ~ scale for 'morl' (approximate)
    periods = scales * sampling_period * 1.03
    return periods, coh, scales
