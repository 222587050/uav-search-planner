"""Üç stratejiyi üç senaryoda ve 1/2/4 İHA ile karşılaştırır.

    python src/run_benchmark.py            # varsayılan: 100 bölüm / konfigürasyon
    python src/run_benchmark.py --episodes 30   # hızlı deneme
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uavsearch.planners import PLANNERS
from uavsearch.sim import run_episode
from uavsearch.world import gaussian_prior, sample_target, uniform_prior

RESULTS = Path(__file__).resolve().parent.parent / "results"
N, R, P_D, T_MAX = 40, 2, 0.9, 400
CENTER, SIGMA = (26, 14), 5.0
OFFSET_CENTER = (9, 30)   # yanlış önsel senaryosunda gerçek hedef buradan gelir


def scenarios():
    plan_gauss = gaussian_prior(N, CENTER, SIGMA)
    return {
        "uniform": (uniform_prior(N), uniform_prior(N)),
        "informative": (plan_gauss, plan_gauss),
        "misspecified": (plan_gauss, gaussian_prior(N, OFFSET_CENTER, SIGMA)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    rows, times = [], {}
    t0 = time.time()
    for sc_i, (sc, (plan, true)) in enumerate(scenarios().items()):
        # Aynı hedef örnekleri tüm strateji ve İHA sayıları için kullanılır (adil karşılaştırma)
        targets = [sample_target(true, np.random.default_rng(1000 * sc_i + e)) for e in range(args.episodes)]
        for K in (1, 2, 4):
            for name, cls in PLANNERS.items():
                steps = []
                for e, tgt in enumerate(targets):
                    rng = np.random.default_rng(7_000_000 + 1000 * e + K)
                    t, _ = run_episode(cls(), plan, tgt, K, R, P_D, T_MAX, rng)
                    steps.append(T_MAX if t is None else t)
                steps = np.array(steps)
                times[(sc, K, name)] = np.sort(steps)
                rows.append({
                    "scenario": sc, "uavs": K, "strategy": name,
                    "success_rate": round(float((steps < T_MAX).mean()), 3),
                    "median_steps": float(np.median(steps)),
                    "p90_steps": float(np.percentile(steps, 90)),
                })
                print(f"{sc:13s} K={K} {name:14s} median={np.median(steps):6.1f}  "
                      f"basari={(steps < T_MAX).mean():.2f}  ({time.time()-t0:.0f}s)", flush=True)

    with open(RESULTS / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Grafik 1: 2 İHA için bulunma süresi dağılımı (CDF)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for a, sc in zip(ax, scenarios()):
        for name in PLANNERS:
            s = times[(sc, 2, name)]
            a.plot(s, np.arange(1, len(s) + 1) / len(s), label=name)
        a.set(title=f"{sc} (2 İHA)", xlabel="Adım")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Hedefin bulunma oranı")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "detection_cdf_2uav.png", dpi=150)
    plt.close(fig)

    # Grafik 2: İHA sayısına göre medyan süre
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for a, sc in zip(ax, scenarios()):
        for name in PLANNERS:
            a.plot([1, 2, 4], [float(np.median(times[(sc, K, name)])) for K in (1, 2, 4)], marker="o", label=name)
        a.set(title=sc, xlabel="İHA sayısı", xticks=[1, 2, 4])
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Medyan bulunma süresi (adım)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "median_vs_uavs.png", dpi=150)
    plt.close(fig)
    print(f"\nKaydedildi: results/benchmark.csv, detection_cdf_2uav.png, median_vs_uavs.png ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
