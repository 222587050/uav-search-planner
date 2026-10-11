import numpy as np

from uavsearch.planners import GreedyBelief, Lawnmower, RandomWalk
from uavsearch.sim import run_episode
from uavsearch.world import gaussian_prior, sample_target, uniform_prior


def test_lawnmower_covers_every_cell_with_perfect_sensor():
    n, r = 30, 2
    covered = np.zeros((n, n), bool)
    plan = uniform_prior(n)
    p = Lawnmower()
    p.reset(n, 2, r, np.random.default_rng(0), plan)
    pos = [(0, 0), (0, 0)]
    for _ in range(3 * n * n):
        pos = p.step(pos, plan)
        for (x, y) in pos:
            covered[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        if covered.all():
            break
    assert covered.all()


def test_planners_keep_agents_inside_grid():
    n = 25
    plan = gaussian_prior(n, (10, 10), 4)
    for cls in (RandomWalk, Lawnmower, GreedyBelief):
        p = cls()
        p.reset(n, 3, 2, np.random.default_rng(1), plan)
        pos = [(0, 0)] * 3
        b = plan.copy()
        for _ in range(200):
            pos = p.step(pos, b)
            assert all(0 <= x < n and 0 <= y < n for x, y in pos)


def test_greedy_beats_random_with_informative_prior():
    n, r = 30, 2
    plan = gaussian_prior(n, (20, 10), 4)
    res = {}
    for cls in (RandomWalk, GreedyBelief):
        steps = []
        for e in range(25):
            tgt = sample_target(plan, np.random.default_rng(e))
            t, _ = run_episode(cls(), plan, tgt, 2, r, 0.9, 300, np.random.default_rng(100 + e))
            steps.append(300 if t is None else t)
        res[cls.__name__] = np.median(steps)
    assert res["GreedyBelief"] < res["RandomWalk"]
