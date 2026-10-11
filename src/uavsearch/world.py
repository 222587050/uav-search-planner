"""Arama alanı, olasılık (belief) haritası ve sensör modeli.

Alan n x n hücrelik bir ızgaradır. Konumlar (x, y), dizi indeksi [y, x].
Hedef (kayıp kişi, araç, ilgi nesnesi) tek bir hücrede, tam yeri bilinmeden durur.
İHA'nın kamera izi: bulunduğu hücre merkezli (2r+1) x (2r+1) kare.
"""
import numpy as np


def uniform_prior(n: int) -> np.ndarray:
    return np.full((n, n), 1.0 / (n * n))


def gaussian_prior(n: int, center, sigma: float) -> np.ndarray:
    ys, xs = np.mgrid[0:n, 0:n]
    d2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
    p = np.exp(-d2 / (2 * sigma**2)) + 1e-9
    return p / p.sum()


def sample_target(prior: np.ndarray, rng: np.random.Generator):
    n = prior.shape[0]
    idx = rng.choice(n * n, p=prior.ravel() / prior.sum())
    return int(idx % n), int(idx // n)  # (x, y)


def footprint_slice(pos, r: int, n: int):
    x, y = pos
    return slice(max(0, y - r), min(n, y + r + 1)), slice(max(0, x - r), min(n, x + r + 1))


def footprint_mass(belief: np.ndarray, r: int) -> np.ndarray:
    """Her hücre için, o hücreye konan bir İHA'nın kamera izindeki toplam olasılık (integral image ile)."""
    n = belief.shape[0]
    k = 2 * r + 1
    P = np.pad(belief, r)
    ii = np.zeros((P.shape[0] + 1, P.shape[1] + 1))
    ii[1:, 1:] = P.cumsum(0).cumsum(1)
    return ii[k:k + n, k:k + n] - ii[0:n, k:k + n] - ii[k:k + n, 0:n] + ii[0:n, 0:n]


def observe(belief: np.ndarray, positions, r: int, p_d: float, target, rng) -> bool:
    """Bir zaman adımı gözlem. Hedef bulunduysa True döner.

    Bulunamadıysa belief, Bayes kuralıyla yerinde güncellenir: taranan hücrelerin
    olasılığı (1 - p_d) ile çarpılır ve yeniden normalize edilir.
    """
    n = belief.shape[0]
    tx, ty = target
    for (x, y) in positions:
        if abs(tx - x) <= r and abs(ty - y) <= r and rng.random() < p_d:
            return True
    for pos in positions:
        sy, sx = footprint_slice(pos, r, n)
        belief[sy, sx] *= (1.0 - p_d)
    total = belief.sum()
    if total <= 0:  # p_d = 1 ve tüm kütle tarandıysa
        belief[:] = 1.0 / belief.size
    else:
        belief /= total
    return False
