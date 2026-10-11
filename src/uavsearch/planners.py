"""Arama stratejileri: rastgele yürüyüş, tarama deseni (lawnmower), belief tabanlı açgözlü planlayıcı."""
import numpy as np

from .world import footprint_mass, footprint_slice


def _identity_belief(self, prior):
    return prior.copy()


def _step_toward(pos, target):
    return (pos[0] + int(np.sign(target[0] - pos[0])), pos[1] + int(np.sign(target[1] - pos[1])))


class RandomWalk:
    name = "random"
    init_belief = _identity_belief

    def reset(self, n, K, r, rng, prior):
        self.n, self.rng = n, rng

    def step(self, positions, belief):
        out = []
        for (x, y) in positions:
            dx, dy = self.rng.integers(-1, 2, size=2)
            out.append((int(np.clip(x + dx, 0, self.n - 1)), int(np.clip(y + dy, 0, self.n - 1))))
        return out


class Lawnmower:
    """Alanı K dikey şeride böler, her İHA kendi şeridini (2r+1) aralıklı satırlarla tarar.
    Tarama bitince başa döner (algılama kaçırma olasılığı varsa ikinci geçiş gerekir)."""
    name = "lawnmower"
    init_belief = _identity_belief

    def reset(self, n, K, r, rng, prior):
        k = 2 * r + 1
        edges = np.linspace(0, n, K + 1).astype(int)
        rows = list(range(r, n, k))
        if rows[-1] < n - 1 - r:
            rows.append(n - 1 - r)
        self.paths, self.idx = [], [0] * K
        for a in range(K):
            x0, x1 = edges[a], edges[a + 1] - 1
            wp = []
            for i, y in enumerate(rows):
                wp += [(x0, y), (x1, y)] if i % 2 == 0 else [(x1, y), (x0, y)]
            self.paths.append(wp)

    def step(self, positions, belief):
        out = []
        for a, pos in enumerate(positions):
            wp = self.paths[a]
            if pos == wp[self.idx[a]]:
                self.idx[a] = (self.idx[a] + 1) % len(wp)
            out.append(_step_toward(pos, wp[self.idx[a]]))
        return out


class GreedyBelief:
    """Her İHA, kamera izindeki olasılık kütlesi / (mesafe + 1) skorunu en çok artıran hücreye gider.
    Çok İHA için sıralı atama: bir İHA hedef seçince o hücrenin izindeki olasılık geçici
    olarak silinir, böylece diğerleri aynı bölgeye yığılmaz."""
    name = "greedy_belief"
    init_belief = _identity_belief

    def reset(self, n, K, r, rng, prior):
        self.n, self.r = n, r
        self.Y, self.X = np.mgrid[0:n, 0:n]

    def step(self, positions, belief):
        b = belief.copy()
        out = []
        for pos in positions:
            G = footprint_mass(b, self.r)
            D = np.maximum(np.abs(self.X - pos[0]), np.abs(self.Y - pos[1]))
            ty, tx = np.unravel_index(np.argmax(G / (D + 1)), G.shape)
            out.append(_step_toward(pos, (int(tx), int(ty))))
            sy, sx = footprint_slice((int(tx), int(ty)), self.r, self.n)
            b[sy, sx] = 0.0
        return out


class GreedyRobust(GreedyBelief):
    """GreedyBelief ile aynı, ama önsel bilgisine tamamen güvenmez:
    başlangıç belief'ine %eps düzgün dağılım karıştırır (yanlış önsele karşı sigorta)."""
    name = "greedy_robust"
    EPS = 0.2

    def init_belief(self, prior):
        n = prior.shape[0]
        return (1.0 - self.EPS) * prior + self.EPS / (n * n)


PLANNERS = {"random": RandomWalk, "lawnmower": Lawnmower,
            "greedy_belief": GreedyBelief, "greedy_robust": GreedyRobust}
