"""Generate the project notebooks from the cell lists below (keeps them diff-able).

    python notebooks/_build_notebooks.py          # writes notebooks/*.ipynb and models/<s>/<s>_analysis.ipynb
    python tools/run_notebook.py notebooks/03_screening.ipynb   # execute one without Jupyter

Every notebook starts with the same setup cell (paths), so each can be run on its own.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = r'''import sys, os, json
from pathlib import Path
ROOT = Path.cwd()
while not (ROOT / "backbone").exists():        # works from notebooks/ or models/<s>/
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from backbone.paths import DATA_ROOT, RUNS_ROOT, FIG_ROOT
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
print("project:", ROOT); print("data:", DATA_ROOT); print("runs:", RUNS_ROOT)'''


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def md(s): return ("markdown", s)
def code(s): return ("code", s)


NOTEBOOKS = {}

NOTEBOOKS["notebooks/00_overview_and_setup.ipynb"] = [
    md("""# MoE_Structures_v2 — overview and setup

Structure-only comparison of routed Transformer structures on synthetic tasks: **Dense, Dense-wide,
MoE, MoH, MoA, MoMHA, MoM, MiTA**, each under three balancing modes (none / load-balance loss /
random warm-up) and with per-structure variants.

| Notebook | What it does |
|---|---|
| 00 | this page: environment, tests |
| 01_data | builds the datasets (E:\\data\\MoE_Structures_v2) and shows examples |
| 02_models | builds every structure and variant: parameters, one forward pass, schematics |
| 03_screening | runs the dataset screening and picks the dataset(s) |
| 04_main_sweep | structures x balancing modes x 3 seeds |
| 05_variants | per-structure variants x balancing modes |
| 06_analysis | all result tables and figures for the report |
| models/<s>/<s>_analysis | everything about one structure |

One trial from the command line: `python -m experiments.run --structure moe --dataset mqar --balance lb --seed 0`
(see docs/code_map.md for the call chain)."""),
    code(SETUP),
    code('''import torch, platform
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("threads", torch.get_num_threads())'''),
    md("## Tests\nParity with the reference code (needs the reference clones, see references/README.md) and the MiTA equivalence test."),
    code('''import subprocess
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True)
print(r.stdout[-1500:])'''),
]

NOTEBOOKS["notebooks/01_data.ipynb"] = [
    md("""# Datasets

Six tasks plus a mixture, all in one vocabulary, written as `.npz` under `DATA_ROOT/<dataset>/`.
Run the build cell once (about a minute); it is skipped when the files exist."""),
    code(SETUP),
    code('''from data_gen.build_all import main as build
if not (DATA_ROOT / "mix" / "manifest.json").exists():
    sys.argv = ["build_all", "--out", str(DATA_ROOT)]; build()
for d in ["mqar", "fact", "mod", "copy", "rev", "logic", "mix"]:
    m = json.loads((DATA_ROOT / d / "manifest.json").read_text())
    print(f"{d:6s}", {s: (v["n"], v["max_len"], round(v["mean_len"], 1)) for s, v in m["splits"].items()})'''),
    md("## One example per task (prompt, then the answers expected at the `<ans>` slots)"),
    code('''from backbone.data import Dataset, load_vocab
V = load_vocab()
for d in ["mqar", "fact", "mod", "copy", "rev", "logic"]:
    ds = Dataset(d, "train")
    L = int(ds.lengths[0]); tok, tgt = ds.tokens[0, :L].tolist(), ds.targets[0, :L].tolist()
    print(f"[{d}]", " ".join(V[t] for t in tok))
    print("   answers:", " ".join(V[t] for t in tgt if t >= 0)); print()'''),
]

NOTEBOOKS["notebooks/02_models.ipynb"] = [
    md("# Structures and variants: sizes, a forward pass, schematics"),
    code(SETUP),
    code('''import torch
from backbone.config import load_run_config, list_variants
from backbone.model import EncoderModel, count_parameters
from backbone.data import Dataset, load_vocab
V = len(load_vocab()); ds = Dataset("mix", "valid")
tok, tgt, pad = ds.batch(torch.arange(32))
rows = []
for s in ["dense", "dense_wide", "moe", "moh", "moa", "momha", "mom", "mita"]:
    for v in list_variants(s):
        cfg = load_run_config(s, v, {"vocab_size": V}); torch.manual_seed(0)
        m = EncoderModel(cfg, V); m.state.record = True
        with torch.no_grad(): logits, aux = m(tok, pad)
        rows.append({"structure": s, "variant": v, "params (no emb/head)": count_parameters(m),
                     "routed modules": len(m.routed_modules())})
display(pd.DataFrame(rows))'''),
    md("## Schematics (drawn by analysis/schematics.py)"),
    code('''from analysis.schematics import draw
import matplotlib.pyplot as plt, matplotlib.image as mpimg
for s in ["dense", "moe", "moh", "moa", "momha", "mom", "mita"]:
    p = draw(s); plt.figure(figsize=(12, 6)); plt.imshow(mpimg.imread(p)); plt.axis("off"); plt.show()'''),
]

NOTEBOOKS["notebooks/03_screening.ipynb"] = [
    md("""# Dataset screening

Every structure (default config, no balancing) on every dataset. A useful dataset spreads the
structures apart by more than seed noise and is neither solved by all nor by none.
Training is resumable: the cell only runs what is missing (each run ~10-20 min on one CPU core)."""),
    code(SETUP),
    code('''RUN = False   # set True to (re)run missing trials here; the sweep can also be run from a terminal
if RUN:
    import subprocess
    subprocess.run([sys.executable, "-m", "experiments.sweep", "--phase", "screen", "--seeds", "0", "--workers", "2"])'''),
    code('''from analysis.collect import load_results, separation
from analysis import figures as F
df = load_results(RUNS_ROOT)
scr = df[(df.variant == "default") & (df.balance == "none")]
display(scr.pivot_table(index="structure", columns="dataset", values="test_acc").round(3))
sep = separation(df); display(sep.round(3))'''),
    code('''import matplotlib.pyplot as plt, matplotlib.image as mpimg
for p in [F.screen_heatmap(df), F.screen_separation(sep)]:
    plt.figure(figsize=(9, 5)); plt.imshow(mpimg.imread(p)); plt.axis("off"); plt.show()'''),
]

NOTEBOOKS["notebooks/04_main_sweep.ipynb"] = [
    md("# Main sweep: structures x balancing modes x 3 seeds"),
    code(SETUP),
    code('''DATASETS = json.loads((ROOT / "configs" / "chosen_datasets.json").read_text())["main"]
print("datasets:", DATASETS)
RUN = False
if RUN:
    import subprocess
    subprocess.run([sys.executable, "-m", "experiments.sweep", "--phase", "main", "--datasets", *DATASETS, "--workers", "2"])'''),
    code('''from analysis.collect import load_results, load_history, load_routing, agg, paired_delta
from analysis.routing import collapse_table
from analysis import figures as F
import matplotlib.pyplot as plt, matplotlib.image as mpimg
def show(p):
    if p: plt.figure(figsize=(10, 5)); plt.imshow(mpimg.imread(p)); plt.axis("off"); plt.show()
df = load_results(RUNS_ROOT); main = df[df.variant == "default"]
for d in DATASETS:
    t = agg(main[main.dataset == d], ["structure", "balance"])
    display(t.pivot(index="structure", columns="balance", values="mean").round(3))
    show(F.main_bars(df, d))'''),
    code('''deltas = pd.concat([paired_delta(df, "lb", "none"), paired_delta(df, "warmup", "none")])
for d in DATASETS:
    display(deltas[(deltas.dataset == d) & (deltas.variant == "default")].round(3))
    show(F.balance_delta(deltas[deltas.variant == "default"], d))'''),
    code('''hist = load_history(RUNS_ROOT); rdf = load_routing(RUNS_ROOT); ct = collapse_table(rdf)
for d in DATASETS:
    for b in ["none", "lb", "warmup"]:
        show(F.curves(hist, d, b))
    show(F.collapse_overview(ct, d))'''),
]

NOTEBOOKS["notebooks/05_variants.ipynb"] = [
    md("# Per-structure variants (docs/variant_registry.md) x balancing modes"),
    code(SETUP),
    code('''DATASETS = json.loads((ROOT / "configs" / "chosen_datasets.json").read_text())["variants"]
RUN = False
if RUN:
    import subprocess
    subprocess.run([sys.executable, "-m", "experiments.sweep", "--phase", "variants", "--datasets", *DATASETS, "--workers", "2"])'''),
    code('''from analysis.collect import load_results, agg
from analysis import figures as F
import matplotlib.pyplot as plt, matplotlib.image as mpimg
df = load_results(RUNS_ROOT)
for d in DATASETS:
    for s in ["moe", "moh", "moa", "momha", "mom", "mita"]:
        t = agg(df[(df.dataset == d) & (df.structure == s)], ["variant", "balance"])
        display(t.pivot(index="variant", columns="balance", values="mean").round(3))
        p = F.variants_bars(df, d, s); plt.figure(figsize=(9, 4)); plt.imshow(mpimg.imread(p)); plt.axis("off"); plt.show()'''),
]

NOTEBOOKS["notebooks/06_analysis.ipynb"] = [
    md("# All tables and figures for the technical report (regenerates reports/figures and reports/tables)"),
    code(SETUP),
    code('''from analysis.report_tables import build_all
paths = build_all()
for p in paths: print(p)'''),
]


def structure_notebook(s, label):
    return [
        md(f"""# {label}: configuration, schematic, results and routing

Code: `models/{s}/` (module + `__init__.py` builder), configs: `models/{s}/configs/*.yaml`.
Deviations from the reference: docs/deviations.md. Variants: docs/variant_registry.md."""),
        code(SETUP),
        code(f'''from backbone.config import list_variants, load_yaml, variant_path
for v in list_variants("{s}"):
    print(f"--- {{v}}"); print(json.dumps(load_yaml(variant_path("{s}", v)).get("structure_params", {{}})))'''),
        code(f'''from analysis.schematics import draw
import matplotlib.pyplot as plt, matplotlib.image as mpimg
def show(p, size=(11, 5.5)):
    if p: plt.figure(figsize=size); plt.imshow(mpimg.imread(p)); plt.axis("off"); plt.show()
show(draw("{s}"))'''),
        code(f'''from analysis.collect import load_results, load_routing, agg
from analysis import figures as F
df = load_results(RUNS_ROOT); d = df[df.structure == "{s}"]
if len(d):
    t = agg(d, ["dataset", "variant", "balance"])
    display(t.pivot_table(index=["dataset", "variant"], columns="balance", values="mean").round(3))
else:
    print("no finished runs yet")'''),
        code(f'''rdf = load_routing(RUNS_ROOT)
if len(rdf) and "{s}" in set(rdf.structure):
    r = rdf[rdf.structure == "{s}"]
    display(r.groupby(["dataset", "variant", "balance"])[["entropy", "max_load", "dead"]].mean().round(3))
    for ds in sorted(set(r.dataset)):
        for v in sorted(set(r[r.dataset == ds].variant)):
            show(F.routing_load(rdf, ds, "{s}", v), (11, 4.5))'''),
    ]


LABELS = {"dense": "Dense", "dense_wide": "Dense-wide", "moe": "MoE", "moh": "MoH", "moa": "MoA", "momha": "MoMHA",
          "mom": "MoM", "mita": "MiTA"}
for s, lab in LABELS.items():
    NOTEBOOKS[f"models/{s}/{s}_analysis.ipynb"] = structure_notebook(s, lab)

if __name__ == "__main__":
    for path, cells in NOTEBOOKS.items():
        p = ROOT / path
        p.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
        print(p)
