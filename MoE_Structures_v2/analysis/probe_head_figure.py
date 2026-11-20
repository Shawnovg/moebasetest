"""Figure for probe 2 (reports/probe/heads2_summary.json) -> reports/figures/probe/head_probe.png"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
rows = json.load(open(ROOT / "reports/probe/heads2_summary.json"))
COL = {"MHA-k": "#7a8699", "MQA-k": "#b9c2cf", "MoA, 8 pool": "#c2477a", "MoA, 16 pool": "#e3a21a", "MoH-block2-SUT": "#2a78d6"}
LS = {"MHA-k": "--", "MQA-k": ":"}
panels = [("acc", "Fact accuracy\n(the model itself)"), ("deliv0", "Layer 1: names readable in\nwhat attention passes on"),
          ("dec_all", "Heads that each hold a name\n(both layers)"), ("cov0", "Layer 1: names whose holder head\nwas switched on")]
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
for ax, (key, t) in zip(axes, panels):
    for f in COL:
        g = defaultdict(list)
        for r in rows:
            if r["f"] == f: g[r["k"]].append(r[key])
        ks = sorted(g)
        ax.plot(ks, [np.mean(g[k]) for k in ks], color=COL[f], ls=LS.get(f, "-"), lw=2.2, marker="o", ms=6, label=f)
    ax.set_title(t, loc="left", fontsize=12, fontweight="bold"); ax.set_xticks([4, 6, 8]); ax.set_xlabel("active heads k")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=9)
fig.savefig(ROOT / "reports/figures/probe/head_probe.png", dpi=160, bbox_inches="tight")
