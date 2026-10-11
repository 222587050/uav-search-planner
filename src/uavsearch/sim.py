"""Tek bir arama senaryosunu çalıştırır."""
import numpy as np

from .world import observe


def run_episode(planner, plan_prior, target, K, r, p_d, t_max, rng, start=(0, 0), record=False):
    """Hedef bulunana veya t_max dolana kadar çalışır.

    Döner: (bulunma adımı veya None, kayıt). plan_prior planlayıcının bildiği önsel;
    gerçek hedef konumu ondan farklı bir dağılımdan gelmiş olabilir.
    """
    n = plan_prior.shape[0]
    belief = planner.init_belief(plan_prior)
    positions = [start] * K
    planner.reset(n, K, r, rng, plan_prior)
    frames = []
    for t in range(1, t_max + 1):
        positions = planner.step(positions, belief)
        found = observe(belief, positions, r, p_d, target, rng)
        if record:
            frames.append((belief.copy(), list(positions)))
        if found:
            return t, frames
    return None, frames
