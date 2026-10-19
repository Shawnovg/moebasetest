"""Generates MoE_Experiment.ipynb from the cell list below (keeps the notebook diff-able).

    python notebooks/_build_notebook.py
"""
import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append(("markdown", s))


def code(s):
    CELLS.append(("code", s))


md(r"""# MoE_Base_Test: experiment notebook

Routed and recurrent Transformer structures on propositional-logic inference (SUT, MoM, and controls).

Every **unit** below is a self-contained cell that defines its functions and then runs a short validation
or produces its output. To use it:

* Run **Unit 0** first. It sets the paths and the budget profile.
* After that, you can run any unit on its own. Units that need earlier results check for them and tell you
  what to run first.
* `PROFILE = "pilot"` reproduces the CPU pilot reported in `reports/Technical_Report.md`.
  `PROFILE = "full"` uses the plan's screening budget (needs a GPU).
* Training is **resumable**. A run whose `result.json` already exists is skipped, so re-running a unit
  only computes what is missing.

| Unit | Responsibility |
|---|---|
| 0 | Setup, budget profiles, plotting style |
| 1 | Environment and upstream-source check |
| 2 | Data preparation and audit (labels, splits, leakage) |
| 3 | Validation tests (unit, parity with upstream code, smoke) |
| 4 | Model inventory: parameters, MAC estimates, measured latency |
| 5 | Experiment plan |
| 6 | Training runner (single run / whole plan) |
| 6b | Follow-up runs chosen from the pilot results |
| 7 | Run status (finished / failed / missing jobs) |
| 8 | Result collection |
| 9 | Statistics (paired seed differences, bootstrap CIs) |
| 10–17 | Figures: one unit per graph |
| 18 | Summary tables for the report |
""")

# ---------------------------------------------------------------------------------------------- 0
md("## Unit 0: Setup")
code(r"""
import sys, os, json, time, math, random, subprocess, copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from moebase import paths

try:
    from IPython.display import display
except ImportError:  # plain-python execution (scripts/run_notebook.py injects its own display)
    if "display" not in globals():
        def display(x):
            print(x.to_string() if hasattr(x, "to_string") else x)

PROFILE = os.environ.get("MOE_PROFILE", "pilot")   # "pilot" (CPU) or "full" (GPU, plan budget)

BUDGETS = {
    # CPU pilot actually executed for the technical report (2 CPU cores, no GPU)
    "pilot": dict(train_limit=None, valid_limit=2000, max_steps=3000, batch_size=64, eval_every=500,
                  warmup_steps=300, seeds_main=[0, 1], seeds_extra=[0], include_reference=False,
                  workers=2, threads_per_worker=1),
    # plan section 11 screening budget
    "full": dict(train_limit=None, valid_limit=None, max_steps=20000, batch_size=128, eval_every=500,
                 warmup_steps=1000, seeds_main=[0, 1, 2, 3, 4], seeds_extra=[0, 1, 2, 3, 4],
                 include_reference=True, workers=1, threads_per_worker=None),
}
B = BUDGETS[PROFILE]
RUNS = paths.RUNS_DIR / PROFILE
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

# plotting style: reference categorical palette (fixed order by family), recessive axes
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY_ORDER = ["Dense", "UT", "Switch", "MoA", "SUT", "SUT fixed", "SUT E=1", "MoM"]
FAMILY_COLOR = dict(zip(FAMILY_ORDER, PALETTE))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})
print("ROOT:", ROOT, "| profile:", PROFILE, "| runs ->", RUNS)
""")

# ---------------------------------------------------------------------------------------------- 1
md("## Unit 1: Environment and upstream sources")
code(r"""
def check_environment():
    import torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.cuda.is_available(), "threads": torch.get_num_threads(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"}
    for name, up in paths.UPSTREAM.items():
        d = paths.upstream_dir(name)
        pin = (d / "PINNED_COMMIT")
        found = pin.read_text().strip() if pin.exists() else None
        if found is None and (d / ".git").exists():
            found = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        info[f"upstream:{name}"] = "OK" if found == up["commit"] else f"MISSING/MISMATCH ({found})"
    return info

ENV = check_environment()
for k, v in ENV.items():
    print(f"{k:28s} {v}")
assert all(v == "OK" for k, v in ENV.items() if k.startswith("upstream:")), "run scripts/fetch_upstream.py"
""")

# ---------------------------------------------------------------------------------------------- 2
md("## Unit 2: Data preparation and audit")
code(r"""
from moebase.data import logic

DATA = paths.DATA_DIR / paths.DATA_VERSION

def prepare_data(force=False):
    if (DATA / "manifest.json").exists() and not force:
        return json.loads((DATA / "manifest.json").read_text())
    raw = logic.load_raw(paths.logic_raw_dir())
    splits = logic.build_splits(raw)
    report = logic.audit(raw, splits, check_labels=True)
    return logic.write_splits(splits, DATA, paths.logic_raw_dir(), report, paths.DATA_VERSION)

def data_summary(man):
    rows = []
    for regime, parts in man["splits"].items():
        for part, d in parts.items():
            rows.append({"regime": regime, "part": part, "n": d["n"], "majority_rate": round(d["majority_label_rate"], 3),
                         "swap_dup": d.get("swap_dup", 0), "max_tokens": d["max_tokens_pair_plus_sep"],
                         "length_bins": f"{min(map(int, d['length_bins']))}-{max(map(int, d['length_bins']))}"})
    return pd.DataFrame(rows)

MAN = prepare_data()
a = MAN["audit"]
print("label mismatches:", a.get("label_mismatches"), "| degenerate clauses:", a.get("degenerate_clauses"),
      "| file-index mismatches:", a["file_index_mismatches"])
print("A/B/C train sizes:", a["systematicity_train_sizes"], "== Ordered Memory Table 2:", a["systematicity_sizes_match_paper"])
assert a.get("label_mismatches", 0) == 0 and a["systematicity_sizes_match_paper"]
display(data_summary(MAN))
""")

# ---------------------------------------------------------------------------------------------- 3
md("## Unit 3: Validation tests\n`fast` runs the unit and parity tests. `slow` adds the tiny-set learning checks (about 1–3 min on CPU).")
code(r"""
def run_tests(which="fast"):
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")]
    if which == "fast":
        args += ["-m", "not slow"]
    elif which == "slow":
        args += ["-m", "slow"]
    r = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
    print(tail)
    return r.returncode

rc = run_tests("fast")
assert rc == 0, "fix failing tests before training"
""")

# ---------------------------------------------------------------------------------------------- 4
md("## Unit 4: Model inventory (parameters, MAC estimates, measured latency)")
code(r"""
import torch
from moebase.train.config import load_config
from moebase.train.build import build_model
from moebase.compute import count_params, sut_block_macs, mom_macs
from moebase.models.sut.model import clear_moe_statistics

CFG_DIR = ROOT / "configs" / "logic"
FAMILY = {"dense6": "Dense", "ut_fixed6": "UT", "ut_sb6": "UT", "switch6": "Switch", "moa6": "MoA",
          "sut_common6": "SUT", "sut_fixed6": "SUT fixed", "sut_e1_6": "SUT E=1",
          "sut_common6_mim_paper": "SUT", "sut_common6_lb": "SUT", "sut_common6_act": "SUT",
          "mom_donor6_cls": "Dense", "mom_ref6_cls": "MoM", "donor6_continue_cls": "Dense",
          "mom_scratch6_cls": "MoM", "sut_ref_seq2seq": "SUT", "mom_donor6_causal": "Dense",
          "mom_ref6_causal": "MoM", "donor6_continue_causal": "Dense", "mom_scratch6_causal": "MoM",
          "sut_common6_long": "SUT", "mom_donor6_cls_full": "Dense", "mom_ref6_cls_fromfull": "MoM",
          "donor6_continue_cls_fromfull": "Dense"}

VARIANT_STYLE = {"sut_common6_mim_paper": "--", "sut_common6_lb": ":", "sut_common6_long": "-.",
                 "donor6_continue_cls": "--", "mom_scratch6_cls": "--", "mom_donor6_cls_full": ":",
                 "mom_ref6_cls_fromfull": ":", "donor6_continue_cls_fromfull": "-."}

def style(c):
    # colour by family (fixed order), line style by variant inside a family
    return FAMILY_COLOR.get(FAMILY.get(c, ""), INK2), VARIANT_STYLE.get(c, "-")

def spread_labels(ax, xs, ys, labels, min_gap_px=11):
    # place labels right of points, pushing apart vertically when they would overlap
    pts = ax.transData.transform(np.c_[xs, ys])
    order = np.argsort(pts[:, 1]); placed = []
    offsets = np.zeros(len(xs))
    for i in order:
        y = pts[i, 1]
        for (px, py) in placed:
            if abs(px - pts[i, 0]) < 160 and abs(py - y) < min_gap_px:
                y = py + min_gap_px
        placed.append((pts[i, 0], y)); offsets[i] = y - pts[i, 1]
    for x, y, t, off in zip(xs, ys, labels, offsets):
        ax.annotate(t, (x, y), xytext=(6, off * 72 / ax.figure.dpi), textcoords="offset points", fontsize=7,
                    color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6) if off else None)

@torch.no_grad()
def model_inventory(names, L=48, batch=64, repeats=5):
    vocab = logic.Vocab()
    rows = []
    for n in names:
        cfg = load_config(CFG_DIR / f"{n}.yaml")
        model, interface, mcfg = build_model(cfg["model"], vocab)
        model.eval()
        kind = cfg["model"]["kind"]
        ids = torch.randint(5, 16, (batch, L))
        args = (ids, torch.full((batch, 2), vocab.eos)) if kind == "sut_seq2seq" else \
               (ids,) if kind.startswith("sut") else (ids, torch.ones(batch, L))
        for _ in range(2):
            model(*args); clear_moe_statistics(model)
        t = []
        for _ in range(repeats):
            t0 = time.perf_counter(); model(*args); t.append(time.perf_counter() - t0); clear_moe_statistics(model)
        if kind.startswith("sut"):
            macs = sut_block_macs(mcfg.block, L, [L] * mcfg.encoder_layers)["total"]
        elif kind.startswith("mom"):
            macs = mom_macs(mcfg, L)["total"]
        else:
            d, f = mcfg.n_embd, mcfg.n_inner
            macs = mcfg.n_layer * (L * (4 * d * d + 2 * d * f) + L * L * d * 2)
        p = count_params(model)
        rows.append({"config": n, "family": FAMILY.get(n, "?"), "params": p["total"], "non_emb_params": p["non_embedding"],
                     "MACs_per_seq_L48 (upper bound)": macs, f"latency_ms_B{batch}_L{L}": 1000 * float(np.median(t))})
    return pd.DataFrame(rows)

INVENTORY_NAMES = ["dense6", "ut_fixed6", "switch6", "moa6", "sut_common6", "sut_fixed6", "sut_e1_6",
                   "mom_donor6_cls", "mom_ref6_cls", "sut_ref_seq2seq"]
INV = model_inventory(INVENTORY_NAMES)
INV.to_csv(RUNS / "inventory.csv", index=False)
display(INV)
""")

# ---------------------------------------------------------------------------------------------- 5
md(r"""## Unit 5: Experiment plan
Panels from the plan, scaled by the profile (`seeds_main` / `seeds_extra`):
* **P1 (main, length regime, `seeds_main`):** Dense, UT, Switch, MoA, SUT, SUT-fixed, SUT E=1, plus SUT with the paper-sign MIM.
* **P4 (MoM two-phase, encoder adaptation, `seeds_main`):** donor → MoM, and the donor trained further for the same budget.
* **P2/P3/P4 extras (`seeds_extra`):** SUT with load balancing; composition split B (Dense, UT, SUT); MoM from scratch.
* **P5 (reference interfaces, when `include_reference`):** SUT seq2seq, and the MoM causal pipeline.

Phase budgets for MoM: donor = ½ budget, phase 2 = ½ budget, continued donor = ½ budget, MoM from scratch = full budget.
The CPU pilot uses 2 main seeds and 1 extra seed. The plan's 3-seed screening needs the `full` profile on a GPU.""")
code(r"""
P1 = ["dense6", "ut_fixed6", "switch6", "moa6", "sut_common6", "sut_fixed6", "sut_e1_6", "sut_common6_mim_paper"]
P2 = ["sut_common6_lb"]   # extra seeds only
P3 = ["dense6", "ut_fixed6", "sut_common6"]

def job(name, seed, regime="length", panel="", base=None, frac=1.0, init=None, convert=None):
    # one training job; base = config file (defaults to name); init = donor job name for phase-2 jobs
    return dict(name=name, base=base or name, seed=seed, regime=regime, panel=panel, frac=frac,
                init=init, convert=convert)

def mom_jobs(seed, sfx, panel, scratch):
    donor = f"mom_donor6_{sfx}"
    s1 = [job(donor, seed, "length", panel, frac=0.5)]
    s2 = [job(f"mom_ref6_{sfx}", seed, "length", panel, frac=0.5, init=donor, convert="mom"),
          job(f"donor6_continue_{sfx}", seed, "length", panel, base=donor, frac=0.5, init=donor, convert=None)]
    if scratch:
        s1.append(job(f"mom_scratch6_{sfx}", seed, "length", panel, base=f"mom_ref6_{sfx}", frac=1.0))
    return s1, s2

def make_plan(budget=B):
    stage1, stage2 = [], []
    for s in budget["seeds_main"]:
        stage1 += [job(c, s, "length", "P1") for c in P1]
        a, b = mom_jobs(s, "cls", "P4", scratch=s in budget["seeds_extra"]); stage1 += a; stage2 += b
    for s in budget["seeds_extra"]:
        stage1.append(job("sut_common6_lb", s, "length", "P2"))
        stage1 += [job(c, s, "sysB", "P3") for c in P3]
    if budget["include_reference"]:
        stage1.append(job("sut_ref_seq2seq", 0, "length", "P5"))
        a, b = mom_jobs(0, "causal", "P5", scratch=True); stage1 += a; stage2 += b
    # longest jobs first so the parallel workers finish together
    cost = {"mom_ref6_cls": 1.6, "mom_scratch6_cls": 1.6, "sut_ref_seq2seq": 3.0}
    stage1.sort(key=lambda j: -cost.get(j["name"], 1.0) * j["frac"])
    return stage1, stage2

STAGE1, STAGE2 = make_plan()
plan_df = pd.DataFrame(STAGE1 + STAGE2)
print(len(plan_df), "training jobs")
display(plan_df.groupby(["panel", "regime"]).size().rename("jobs"))
""")

# ---------------------------------------------------------------------------------------------- 6
md("## Unit 6: Training runner\nEach job becomes a frozen `config.json` in its run folder and runs `scripts/train.py` in a subprocess. Up to `workers` jobs run in parallel. Finished jobs (`result.json` present) are skipped.")
code(r"""
from concurrent.futures import ThreadPoolExecutor
from moebase.train.config import load_config
from moebase.train.loop import run as train_run

def run_dir(name, seed, regime):
    return RUNS / regime / f"{name}_s{seed}"

def budget_config(j, budget=B):
    cfg = load_config(CFG_DIR / f"{j['base']}.yaml")
    cfg["name"] = j["name"]
    steps = max(1, int(budget["max_steps"] * j["frac"]))
    cfg["train"].update(seed=j["seed"], max_steps=steps, batch_size=budget["batch_size"],
                        eval_every=min(budget["eval_every"], steps), log_every=50)
    cfg["data"].update(regime=j["regime"], train_limit=budget["train_limit"], valid_limit=budget["valid_limit"])
    o = cfg["optim"]
    if o.get("schedule") == "inverse_sqrt":   # fairseq warm-up shortened to fit the budget (logged deviation)
        o["warmup_steps"] = min(o.get("warmup_steps", 4000), max(budget["warmup_steps"], steps // 4))
    elif "warmup_ratio" not in o:
        o["warmup_steps"] = min(budget["warmup_steps"], steps // 5)
    if j["init"]:
        cfg["init_from"] = {"checkpoint": str(run_dir(j["init"], j["seed"], j["regime"]) / "best.pt"),
                            "convert": j["convert"]}
    elif "init_from" in cfg:
        cfg["init_from"] = None
    return cfg

def run_one(j, budget=B, force=False):
    out = run_dir(j["name"], j["seed"], j["regime"])
    if (out / "result.json").exists() and not force:
        return "skip"
    out.mkdir(parents=True, exist_ok=True)
    cfg = budget_config(j, budget)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    if budget.get("threads_per_worker"):
        env["OMP_NUM_THREADS"] = str(budget["threads_per_worker"])
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), str(out / "config.json"), "--out", str(out)],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    (out / "stdout.txt").write_text(r.stdout + r.stderr)
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"{status:6s} {j['regime']:6s} {j['name']:24s} s{j['seed']}  {time.time() - t0:6.0f}s  " +
          (r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else (r.stderr.strip().splitlines() or [''])[-1]),
          flush=True)
    return status

def run_plan(jobs, budget=B, force=False):
    with ThreadPoolExecutor(max_workers=budget.get("workers", 1)) as ex:
        return list(ex.map(lambda j: run_one(j, budget, force), jobs))

# validation of the unit: a 20-step job must finish and write a result
_check = job("dense6", 0, "length", "check")
_b = dict(B, max_steps=20, eval_every=20, valid_limit=200)
_cfg = budget_config(_check, _b); _cfg["data"]["train_limit"] = 500
_res = train_run(_cfg, out_dir=RUNS / "_unit_check", evaluate_tests=False)
assert 0 <= _res["valid"]["acc"] <= 1
print("runner OK; valid acc after 20 steps:", round(_res["valid"]["acc"], 3))
""")
code(r"""
# Execute the plan (resumable). Stage 2 = MoM phase-2 jobs that need the stage-1 donors.
t0 = time.time()
run_plan(STAGE1)
run_plan(STAGE2)
print(f"plan finished in {(time.time() - t0) / 3600:.2f} h")
""")

md(r"""## Unit 6b: Follow-up runs chosen from the pilot (P6)
The pilot showed two things a fixed 3,000-step budget cannot settle:
1. The MoM donor (GPT-2 style, ½ budget = 1,500 steps) never left the majority-class plateau. So MoM-from-donor tested nothing. **F1** trains the donor for the full budget, then runs MoM and the continued-donor control from it.
2. Weight-shared and sparse models learned more slowly per step. **F2** gives SUT 3× the budget (9,000 steps, seed 0) to separate slow learning from failure to learn.""")
code(r"""
FOLLOWUP1 = [job("sut_common6_long", 0, "length", "P6", base="sut_common6", frac=3.0),
             job("mom_donor6_cls_full", 0, "length", "P6", base="mom_donor6_cls", frac=1.0)]
FOLLOWUP2 = [job("mom_ref6_cls_fromfull", 0, "length", "P6", base="mom_ref6_cls", frac=0.5,
                 init="mom_donor6_cls_full", convert="mom"),
             job("donor6_continue_cls_fromfull", 0, "length", "P6", base="mom_donor6_cls", frac=0.5,
                 init="mom_donor6_cls_full", convert=None)]
if PROFILE == "pilot":
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_long = ex.submit(run_one, FOLLOWUP1[0])
        run_one(FOLLOWUP1[1]); run_plan(FOLLOWUP2)   # donor, then both phase-2 jobs, on the other worker
        f_long.result()
    print(f"follow-ups finished in {(time.time() - t0) / 60:.0f} min")
""")

# ---------------------------------------------------------------------------------------------- 7
md("## Unit 7: Run status\nWhich jobs finished, failed or are missing; training time per job.")
code(r"""
def run_status(jobs):
    rows = []
    for j in jobs:
        d = run_dir(j["name"], j["seed"], j["regime"])
        r = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else None
        rows.append(dict(panel=j["panel"], regime=j["regime"], name=j["name"], seed=j["seed"],
                         status="done" if r else ("failed" if (d / "stdout.txt").exists() else "missing"),
                         train_min=round(r["train_time_s"] / 60, 1) if r else None,
                         best_step=r["best_step"] if r else None))
    return pd.DataFrame(rows)

STATUS = run_status(STAGE1 + STAGE2 + (FOLLOWUP1 + FOLLOWUP2 if PROFILE == "pilot" else []))
display(STATUS.groupby("status").size())
display(STATUS[STATUS.status != "done"])
""")

# ---------------------------------------------------------------------------------------------- 8
md("## Unit 8: Result collection\nOne row per (run, evaluation partition). Saved to `runs/<profile>/results_long.csv`.")
code(r"""
DISPLAY = {"dense6": "Dense", "ut_fixed6": "UT (fixed)", "switch6": "Switch", "moa6": "MoA",
           "sut_common6": "SUT", "sut_fixed6": "SUT fixed-depth", "sut_e1_6": "SUT E=1",
           "sut_common6_mim_paper": "SUT, MIM paper sign", "sut_common6_lb": "SUT, load balancing",
           "mom_donor6_cls": "MoM donor (dense)", "mom_ref6_cls": "MoM (from donor)",
           "donor6_continue_cls": "Donor continued", "mom_scratch6_cls": "MoM (scratch)",
           "sut_ref_seq2seq": "SUT reference (seq2seq)", "mom_donor6_causal": "Donor (causal)",
           "mom_ref6_causal": "MoM reference (causal)", "donor6_continue_causal": "Donor continued (causal)",
           "mom_scratch6_causal": "MoM scratch (causal)", "sut_common6_long": "SUT, 3x budget",
           "mom_donor6_cls_full": "MoM donor, full budget", "mom_ref6_cls_fromfull": "MoM from full donor",
           "donor6_continue_cls_fromfull": "Full donor continued"}

def collect_results(root=RUNS):
    rows, curves = [], []
    for res_path in sorted(root.glob("*/*/result.json")):
        regime, run = res_path.parent.parent.name, res_path.parent.name
        name, seed = run.rsplit("_s", 1)
        r = json.loads(res_path.read_text())
        rec = json.loads((res_path.parent / "record.json").read_text())
        base = {"config": name, "label": DISPLAY.get(name, name), "family": FAMILY.get(name, "?"),
                "seed": int(seed), "regime": regime, "params": rec["n_params"], "best_step": r["best_step"],
                "train_time_s": r["train_time_s"]}
        for part, m in [("valid", r["valid"])] + list(r.get("test", {}).items()):
            row = dict(base, part=part, acc=m["acc"], brier=m["brier"], macro_f1=m["macro_f1"], nll=m["nll"],
                       ece=m["ece"], n=m["n"], acc_nodup=m.get("acc_excl_swap_dup", m["acc"]),
                       expected_depth=m.get("expected_depth"), invalid_rate=m.get("invalid_rate"))
            rows.append(row)
        for line in (res_path.parent / "log.jsonl").read_text().splitlines():
            e = json.loads(line)
            if "valid" in e:
                curves.append(dict(config=name, seed=int(seed), regime=regime, step=e["step"],
                                   valid_acc=e["valid"]["acc"], valid_brier=e["valid"]["brier"]))
    return pd.DataFrame(rows), pd.DataFrame(curves)

DF, CURVES = collect_results()
DF.to_csv(RUNS / "results_long.csv", index=False); CURVES.to_csv(RUNS / "curves.csv", index=False)
print(len(DF), "rows from", DF[["config", "seed", "regime"]].drop_duplicates().shape[0], "runs")
display(DF[DF.part != "valid"].pivot_table(index="label", columns=["regime", "part"], values="acc", aggfunc="mean").round(3))
""")

# ---------------------------------------------------------------------------------------------- 9
md("## Unit 9: Statistics\nPaired by seed against a baseline: mean difference, t-interval, and percentile bootstrap over seeds. With 3 seeds these intervals are wide; that is the honest answer.")
code(r"""
from scipy import stats as st

def paired_differences(df, baseline, configs, regime="length", part="valid", metric="acc", n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    sub = df[(df.regime == regime) & (df.part == part)]
    b = sub[sub.config == baseline].set_index("seed")[metric]
    rows = []
    for c in configs:
        x = sub[sub.config == c].set_index("seed")[metric]
        common = sorted(set(x.index) & set(b.index))
        if not common or c == baseline:
            continue
        d = (x.loc[common] - b.loc[common]).values
        n = len(d)
        mean = d.mean()
        sd = d.std(ddof=1) if n > 1 else float("nan")
        tci = st.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
        boots = rng.choice(d, size=(n_boot, n), replace=True).mean(1)
        rows.append({"config": c, "label": DISPLAY.get(c, c), "vs": DISPLAY.get(baseline, baseline), "n_seeds": n,
                     "mean_diff": mean, "t95_low": mean - tci, "t95_high": mean + tci,
                     "boot95_low": np.percentile(boots, 2.5), "boot95_high": np.percentile(boots, 97.5),
                     "wins": int((d > 0).sum())})
    return pd.DataFrame(rows)

def seed_table(df, configs, regime="length", parts=("valid", "test_iid", "test_len"), metric="acc"):
    sub = df[(df.regime == regime) & df.config.isin(configs) & df.part.isin(parts)]
    g = sub.groupby(["label", "part"])[metric].agg(["mean", "std", "count"]).unstack("part")
    return g

if len(DF):
    STATS_VALID = paired_differences(DF, "dense6", P1 + P2, part="valid")
    STATS_LEN = paired_differences(DF, "dense6", P1 + P2, part="test_len")
    STATS_UT = paired_differences(DF, "ut_fixed6", ["sut_fixed6", "sut_common6", "sut_e1_6"], part="valid")
    STATS_LEN_UT = paired_differences(DF, "ut_fixed6", ["sut_fixed6", "sut_common6", "sut_e1_6"], part="test_len")
    for t, n in [(STATS_VALID, "valid vs Dense"), (STATS_LEN, "length test vs Dense"),
                 (STATS_UT, "valid vs UT"), (STATS_LEN_UT, "length test vs UT")]:
        print("\n##", n); display(t.round(4))
""")

# ---------------------------------------------------------------------------------------------- figures
md("## Unit 10: Figure: validation accuracy and Brier per model (seed dots + mean)")
code(r"""
def fig_validation_overview(df, configs, regime="length", fname="fig1_validation_overview.png"):
    sub = df[(df.regime == regime) & (df.part == "valid") & df.config.isin(configs)]
    order = [c for c in configs if c in set(sub.config)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(order) + 1.2), sharey=True)
    for ax, metric, title in [(axes[0], "acc", "Validation accuracy (higher is better)"),
                              (axes[1], "brier", "Validation Brier score (selection metric, lower is better)")]:
        for i, c in enumerate(order):
            v = sub[sub.config == c][metric].values
            col = FAMILY_COLOR.get(FAMILY.get(c, ""), INK2)
            ax.scatter(v, [i] * len(v), s=22, color=col, alpha=0.55, zorder=3, linewidths=0)
            ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=col, lw=2.5, zorder=4)
            ax.annotate(f"{v.mean():.3f}", (v.max(), i), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_yticks(range(len(order)), [DISPLAY.get(c, c) for c in order])
        ax.invert_yaxis()
    majority = MAN["splits"]["length"]["valid"]["majority_label_rate"]
    axes[0].axvline(majority, color=INK2, ls=":", lw=1); axes[0].text(majority, -0.7, " majority class", fontsize=7, color=INK2)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_validation_overview(DF, P1 + P2 + ["sut_common6_long", "mom_donor6_cls", "mom_ref6_cls", "donor6_continue_cls", "mom_scratch6_cls",
                            "mom_donor6_cls_full", "mom_ref6_cls_fromfull", "donor6_continue_cls_fromfull"])
""")

md("## Unit 11: Figure: accuracy by number of operators (in-distribution 0–6, length generalisation 7–12)")
code(r"""
def per_length_frame(root=RUNS, regime="length", parts=("test_iid", "test_len")):
    rows = []
    for res_path in sorted((root / regime).glob("*/result.json")):
        name, seed = res_path.parent.name.rsplit("_s", 1)
        r = json.loads(res_path.read_text())
        for part in parts:
            for L, m in r.get("test", {}).get(part, {}).get("per_length", {}).items():
                rows.append(dict(config=name, seed=int(seed), part=part, ops=int(L), acc=m["acc"], n=m["n"],
                                 expected_depth=m.get("expected_depth")))
    return pd.DataFrame(rows)

PL = per_length_frame()

def fig_accuracy_by_length(pl, groups, fname="fig2_accuracy_by_length.png"):
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, configs) in zip(axes, groups.items()):
        for c in configs:
            s = pl[pl.config == c]
            if s.empty:
                continue
            g = s.groupby("ops")["acc"].agg(["mean", "min", "max"])
            col, ls = style(c)
            ax.plot(g.index, g["mean"], marker="o", ms=3.5, color=col, ls=ls, label=DISPLAY.get(c, c))
            ax.fill_between(g.index, g["min"], g["max"], color=col, alpha=0.12, lw=0)
        ax.axvline(6.5, color=INK2, ls="--", lw=1)
        ax.text(6.6, 0.33, "unseen\nlengths →", fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left"); ax.set_xlabel("number of operators (max over the two clauses)")
        ax.set_ylim(0.3, 1.01); ax.legend(fontsize=7.5, loc="lower left")
    axes[0].set_ylabel("test accuracy (line = seed mean, band = seed range)")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_accuracy_by_length(PL, {"Controls vs SUT": ["dense6", "ut_fixed6", "switch6", "moa6", "sut_common6"],
                            "SUT mechanism ablations": ["ut_fixed6", "sut_fixed6", "sut_e1_6", "sut_common6"],
                            "MoM two-phase (encoder adaptation)": ["mom_donor6_cls", "donor6_continue_cls", "mom_ref6_cls", "mom_scratch6_cls",
                                                                   "mom_donor6_cls_full", "donor6_continue_cls_fromfull", "mom_ref6_cls_fromfull"]})
""")

md("## Unit 12: Figure: compositional generalisation (sysB: excluded `( and ( not * ) )`)")
code(r"""
def fig_composition(df, configs, regime="sysB", fname="fig3_composition_sysB.png"):
    parts = [("test_iid", "in-distribution"), ("test_comp_seenlen", "held-out pattern,\nseen lengths"),
             ("test_comp_newlen", "held-out pattern,\nunseen lengths")]
    sub = df[(df.regime == regime) & df.config.isin(configs)]
    if sub.empty:
        print("no sysB runs yet"); return
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.8 / len(configs)
    for j, c in enumerate(configs):
        col = FAMILY_COLOR.get(FAMILY.get(c, ""), INK2)
        for i, (p, _) in enumerate(parts):
            v = sub[(sub.config == c) & (sub.part == p)]["acc"].values
            if len(v) == 0:
                continue
            x = i + (j - (len(configs) - 1) / 2) * w
            ax.bar(x, v.mean(), width=w - 0.03, color=col, label=DISPLAY.get(c, c) if i == 0 else None, zorder=3)
            ax.scatter([x] * len(v), v, s=10, color=INK, zorder=4, linewidths=0)
            ax.text(x, v.mean() + 0.01, f"{v.mean():.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(parts)), [p[1] for p in parts]); ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("accuracy"); ax.set_title("Composition split B (bars = seed mean, dots = seeds)", loc="left")
    ax.legend(fontsize=7.5, ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_composition(DF, P3)
""")

md("## Unit 13: Figure: SUT halting depth versus expression length")
code(r"""
def fig_expected_depth(pl, configs=("sut_common6", "sut_common6_mim_paper", "sut_common6_lb"), fname="fig4_halting_depth.png"):
    sub = pl[pl.config.isin(configs) & pl.expected_depth.notna()]
    if sub.empty:
        print("no halting runs"); return
    fig, ax = plt.subplots(figsize=(6, 3.3))
    for i, c in enumerate(configs):
        s = sub[sub.config == c]
        if s.empty:
            continue
        g = s.groupby("ops")["expected_depth"].agg(["mean", "min", "max"])
        col = PALETTE[[4, 6, 1][i % 3]]
        ax.plot(g.index, g["mean"], marker="o", ms=4, color=col, label=DISPLAY.get(c, c))
        ax.fill_between(g.index, g["min"], g["max"], color=col, alpha=0.12, lw=0)
    ax.axvline(6.5, color=INK2, ls="--", lw=1)
    ax.set_xlabel("number of operators"); ax.set_ylabel("expected halting index per token (0-based)")
    ax.set_title("Does SUT compute longer on longer expressions? (cap = 5)", loc="left"); ax.legend(fontsize=7.5, loc="center right")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_expected_depth(PL)
""")

md("## Unit 14: Figure: expert and module usage (routing balance)")
code(r"""
def routing_usage(root=RUNS, configs=("sut_common6", "sut_common6_mim_paper", "sut_common6_lb", "switch6", "moa6", "mom_ref6_cls"), seed=0):
    out = {}
    for c in configs:
        p = root / "length" / f"{c}_s{seed}" / "result.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())["routing"]
        u = r["sut_expert_usage"] or r["mom_module_usage"]
        out[c] = {k: v for k, v in u.items() if len(v) > 1}
    return out

def usage_entropy(frac):
    p = np.asarray(frac, float); p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(frac)))

def fig_routing_usage(usage, fname="fig5_routing_usage.png"):
    rows = []
    for c, mods in usage.items():
        for m, frac in mods.items():
            short = m.replace("encoder.", "").replace("universal_layer.", "").replace("transformer.h.1.", "")
            short = short.replace("self_attn.q_proj", "attn").replace("moe", "ffn").replace("layers.", "L")
            rows.append((DISPLAY.get(c, c), short, frac))
    if not rows:
        print("no routing data"); return
    kmax = max(len(f) for _, _, f in rows)
    M = np.full((len(rows), kmax), np.nan)
    for i, (_, _, f) in enumerate(rows):
        M[i, :len(f)] = f
    fig, ax = plt.subplots(figsize=(8.2, 0.24 * len(rows) + 1.3))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(rows)), [f"{a} | {b}  (H={usage_entropy(f):.2f})" for a, b, f in rows], fontsize=7)
    ax.set_xticks(range(kmax), [str(i) if i < kmax - 1 or not any("step" in r[1] for r in rows) else f"{i}" for i in range(kmax)])
    ax.set_xlabel("expert / module index (MoM: last index = SKIP)"); ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.03, label="share of routed slots")
    ax.set_title("Routing usage on validation, seed 0 (H: normalised entropy, 1 = uniform)", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

USAGE = routing_usage()
fig_routing_usage(USAGE)
""")

md("## Unit 15: Figure: validation learning curves")
code(r"""
def fig_training_curves(curves, configs, regime="length", fname="fig6_learning_curves.png", offsets=None, title=None):
    # offsets: shift a phase-2 run's steps by its donor's steps so the x axis is total training steps
    sub = curves[(curves.regime == regime) & curves.config.isin(configs)]
    if sub.empty:
        print("no curves"); return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for c in configs:
        s = sub[sub.config == c]
        if s.empty:
            continue
        g = s.groupby("step")["valid_acc"].mean()
        col, ls = style(c)
        ax.plot(g.index + (offsets or {}).get(c, 0), g.values, color=col, ls=ls, label=DISPLAY.get(c, c))
    ax.axhline(MAN["splits"]["length"]["valid"]["majority_label_rate"], color=INK2, ls=":", lw=1)
    ax.set_xlabel("total training steps"); ax.set_ylabel("validation accuracy (seed mean)")
    ax.set_title(title or "Learning curves under the pilot budget", loc="left"); ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_training_curves(CURVES, P1)
fig_training_curves(CURVES, ["dense6", "sut_common6", "sut_common6_long", "mom_donor6_cls_full", "mom_ref6_cls_fromfull",
                             "donor6_continue_cls_fromfull"], fname="fig6b_followup_curves.png",
                    offsets={"mom_ref6_cls_fromfull": B["max_steps"], "donor6_continue_cls_fromfull": B["max_steps"]},
                    title="Follow-ups: longer SUT run and MoM from a fully trained donor")
""")

md("## Unit 16: Figure: accuracy versus cost (parameters and measured latency)")
code(r"""
def fig_cost_accuracy(df, inv, part="test_len", fname="fig7_cost_vs_accuracy.png"):
    acc = df[(df.regime == "length") & (df.part == part)].groupby("config")["acc"].mean()
    j = inv.set_index("config").join(acc, how="inner")
    if j.empty:
        print("nothing to plot"); return
    lat = [c for c in inv.columns if c.startswith("latency")][0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, x, xl in [(axes[0], "params", "trainable parameters"), (axes[1], lat, "measured CPU latency (ms, batch 64, L=48)")]:
        for c, r in j.iterrows():
            ax.scatter(r[x], r["acc"], s=46, color=style(c)[0], zorder=3, edgecolor="#fcfcfb", linewidth=1.5)
        ax.set_xlabel(xl)
        fig.canvas.draw()
        spread_labels(ax, j[x].values, j["acc"].values, [DISPLAY.get(c, c) for c in j.index])
    axes[0].set_ylabel(f"{part} accuracy (seed mean)")
    axes[0].set_title("Accuracy versus size and measured cost", loc="left")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_cost_accuracy(DF, INV)
""")

md("## Unit 17: Figure: confusion matrices (length-generalisation test, seed 0)")
code(r"""
def fig_confusion(configs, regime="length", part="test_len", seed=0, fname="fig8_confusion.png"):
    labs = logic.LABELS
    fig, axes = plt.subplots(1, len(configs), figsize=(3.3 * len(configs), 3.3))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, configs):
        p = RUNS / regime / f"{c}_s{seed}" / "result.json"
        if not p.exists():
            ax.set_visible(False); continue
        cm = np.array(json.loads(p.read_text())["test"][part]["confusion"], float)
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        ax.set_xticks(range(7), labs); ax.set_yticks(range(7), labs)
        for i in range(7):
            for k in range(7):
                if cmn[i, k] >= 0.05:
                    ax.text(k, i, f"{cmn[i, k]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if cmn[i, k] > 0.55 else INK)
        ax.set_title(DISPLAY.get(c, c), loc="left"); ax.set_xlabel("predicted")
    axes[0].set_ylabel("true relation")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_confusion(["dense6", "ut_fixed6", "sut_common6", "mom_ref6_cls"])
""")

md("## Unit 18: Summary tables for the report")
code(r"""
def summary_tables(df):
    out = {}
    m = df[df.regime == "length"].pivot_table(index="label", columns="part", values="acc", aggfunc=["mean", "std"])
    out["length_regime"] = m
    comp = df[df.regime == "sysB"].pivot_table(index="label", columns="part", values="acc", aggfunc=["mean", "std"])
    out["composition_sysB"] = comp
    extra = df[(df.regime == "length") & (df.part == "test_iid")].groupby("label")[["acc_nodup", "brier", "macro_f1", "ece"]].mean()
    out["iid_extra"] = extra
    depth = df[(df.regime == "length") & (df.part == "test_len") & df.expected_depth.notna()].groupby("label")["expected_depth"].mean()
    out["depth"] = depth
    return out

TABLES = summary_tables(DF)
for k, t in TABLES.items():
    print("\n##", k); display(t.round(4))
SUMMARY = {k: json.loads(t.round(5).to_json()) for k, t in TABLES.items()}
(RUNS / "summary_tables.json").write_text(json.dumps(SUMMARY, indent=1))
""")


def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (t, s) in enumerate(CELLS):
        cell = {"cell_type": t, "id": f"c{i:02d}", "metadata": {}, "source": s.strip("\n").splitlines(True)}
        if t == "code":
            cell.update(outputs=[], execution_count=None)
        nb["cells"].append(cell)
    Path(path).write_text(json.dumps(nb, indent=1), encoding="utf-8")


if __name__ == "__main__":
    build(Path(__file__).with_name("MoE_Experiment.ipynb"))
    print("written", len(CELLS), "cells")
