import numpy as np

from uavsearch.world import footprint_mass, gaussian_prior, observe, uniform_prior


def test_priors_are_normalized():
    assert abs(uniform_prior(20).sum() - 1) < 1e-12
    assert abs(gaussian_prior(20, (5, 5), 3).sum() - 1) < 1e-12


def test_footprint_mass_matches_brute_force():
    rng = np.random.default_rng(0)
    b = rng.random((15, 15)); b /= b.sum()
    r = 2
    G = footprint_mass(b, r)
    for (x, y) in [(0, 0), (7, 7), (14, 3), (3, 14)]:
        expect = b[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].sum()
        assert abs(G[y, x] - expect) < 1e-12


def test_certain_detection_when_target_in_footprint():
    b = uniform_prior(20)
    assert observe(b, [(5, 5)], 2, 1.0, (6, 4), np.random.default_rng(0)) is True


def test_no_detection_lowers_scanned_belief_and_stays_normalized():
    b = uniform_prior(20)
    before = b[5, 5]
    found = observe(b, [(5, 5)], 2, 0.9, (15, 15), np.random.default_rng(0))
    assert found is False
    assert b[5, 5] < before
    assert abs(b.sum() - 1) < 1e-12
