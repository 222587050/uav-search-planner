"""Yanlış önsel senaryosunda 2 İHA'lık greedy_robust aramasını GIF'e çevirir.

    python src/animate.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uavsearch.planners import GreedyRobust
from uavsearch.sim import run_episode
from uavsearch.world import gaussian_prior, sample_target

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
N, R, P_D, K = 40, 2, 0.9, 2
plan = gaussian_prior(N, (26, 14), 5.0)       # planlayıcının bildiği (yanlış) önsel
true = gaussian_prior(N, (9, 30), 5.0)        # hedefin gerçekte geldiği dağılım

for seed in range(200):
    target = sample_target(true, np.random.default_rng(seed))
    t, frames = run_episode(GreedyRobust(), plan, target, K, R, P_D, 400, np.random.default_rng(seed), record=True)
    if t is not None and 70 <= t <= 160:
        break
print(f"Gösterilen bölüm: seed={seed}, hedef={target}, bulunma adımı={t}")

fig, ax = plt.subplots(figsize=(5.2, 5.2))
im = ax.imshow(frames[0][0], origin="lower", cmap="magma", vmin=0, vmax=0.01)
ax.plot(*target, marker="*", color="cyan", ms=14, ls="", label="hedef")
colors = ["lime", "deepskyblue"]
boxes = [ax.add_patch(Rectangle((0, 0), 2 * R + 1, 2 * R + 1, fill=False, ec=colors[a], lw=1.5)) for a in range(K)]
dots = [ax.plot([], [], "o", color=colors[a], ms=6)[0] for a in range(K)]
title = ax.set_title("")
ax.set(xticks=[], yticks=[])


def draw(i):
    belief, pos = frames[i]
    im.set_data(belief)
    for a, (x, y) in enumerate(pos):
        dots[a].set_data([x], [y])
        boxes[a].set_xy((x - R - 0.5, y - R - 0.5))
    title.set_text(f"Adım {i + 1}" + ("  — hedef bulundu" if i == len(frames) - 1 else ""))
    return [im, title, *dots, *boxes]


order = list(range(0, len(frames), 2)) + [len(frames) - 1] * 8
FuncAnimation(fig, lambda k: draw(order[k]), frames=len(order)).save(RESULTS / "search_demo.gif", writer=PillowWriter(fps=12))
print("Kaydedildi: results/search_demo.gif")
