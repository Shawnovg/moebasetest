"""Generates MoE_Analysis.ipynb (mechanistic analysis of the trained pilot models).

    python notebooks/_build_analysis_notebook.py
    python scripts/run_notebook.py notebooks/MoE_Analysis.ipynb --out notebooks/MoE_Analysis_executed.ipynb

Every compute unit caches its results under runs/analysis/, so a unit can be re-run on its own and the
figure units only read those caches.
"""
import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append(("markdown", s))


def code(s):
    CELLS.append(("code", s))


md(r"""# MoE_Base_Test: mechanistic analysis of the trained models

The pilot showed that the dense Transformer beats SUT and MoM on propositional-logic inference. This notebook
looks **inside** the trained checkpoints to explain why. It runs a series of experiments on the hidden state
after every block application (every layer for untied models, every recurrent step for UT/SUT, every assembly
step for MoM):

| Unit | Experiment | Question it answers |
|---|---|---|
| 0–1 | Setup, model loading, checkpoint sanity check | Do the reloaded models reproduce their logged accuracy? |
| 2 | Semantic targets: region bits, truth tables, syntax | What is there to decode? (validated: 4 region bits determine the label) |
| 3 | Minimal pairs for patching | Single-token edits that flip the label |
| 4 | State capture with step hooks | Hidden states per step at `<cls>` and pooled over each clause |
| 5 | **Linear probing** (+ random-init control, OOD lengths) | At which step does each model *represent* the clause meanings and their relation? |
| 6 | **Activation patching** | Where and when does the decisive information move to `<cls>`? |
| 7 | **Sparse autoencoder** on the `<cls>` readout | Are the final features organised by logical concepts? |
| 8 | Routing analysis | What do the routers condition on (token type, step)? |
| 9–12 | **Ablations** in the authors' style: depth truncation, halting threshold, expert knockout and random routing, MoM module knockout / forced SKIP | Which units does each model actually use? |
| 13 | Training-time ablations (from the pilot runs) | MIM sign, load balancing, fixed depth, E=1, budget |
| 14 | Synthesis | Why SUT and MoM fall behind the dense model |

Figures are one unit each (5a, 5b, …). Each compute unit saves its result to `runs/analysis/` and each figure
unit reads only from there, so you can re-draw any figure without recomputing.

**Notation.** For a pair of clauses $A, B$ over 6 variables there are $2^6 = 64$ worlds. With
$S_A, S_B \subseteq \{0,\dots,63\}$ the satisfying worlds, the four *region bits* are
$r_1 = [S_A \cap S_B \neq \varnothing]$, $r_2 = [S_A \setminus S_B \neq \varnothing]$,
$r_3 = [S_B \setminus S_A \neq \varnothing]$, $r_4 = [\overline{S_A \cup S_B} \neq \varnothing]$, and the label
is a fixed function of $(r_1, r_2, r_3, r_4)$ (Unit 2 checks this on every example).
""")

# ---------------------------------------------------------------------------------------------- 0
md("## Unit 0: Setup")
code(r'''
import sys, os, json, time, math, random, copy, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from moebase.data import logic
from moebase.data.logic import Vocab, Example
from moebase.train.build import build_model

try:
    from IPython.display import display
except ImportError:
    if "display" not in globals():
        def display(x):
            print(x.to_string() if hasattr(x, "to_string") else x)

PROFILE = os.environ.get("MOE_ANALYSIS_PROFILE", "quick")   # "quick" (CPU, used for the report) or "full"
SIZES = {
    "quick": dict(n_train=8000, n_test=3000, n_long=2000, n_pairs=500, n_ablate=2000, sae_steps=4000),
    "full": dict(n_train=30000, n_test=10000, n_long=6000, n_pairs=3000, n_ablate=10000, sae_steps=20000),
}
S = SIZES[PROFILE]
torch.set_num_threads(int(os.environ.get("MOE_THREADS", "2")))

RUN_ROOT = ROOT / "runs" / "pilot" / "length"          # checkpoints from MoE_Experiment.ipynb (pilot profile)
DATA = ROOT / "data" / "logic-v1" / "length"
OUT = ROOT / "runs" / "analysis" / PROFILE
FIG = ROOT / "reports" / "figures" / "analysis"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

# models analysed (seed 0 of each pilot condition); key = run name without the seed suffix
MODELS = {
    "dense6":                dict(label="Dense",                 family="Dense"),
    "ut_fixed6":             dict(label="UT (fixed depth)",      family="UT"),
    "switch6":               dict(label="Switch",                family="Switch"),
    "moa6":                  dict(label="MoA",                   family="MoA"),
    "sut_fixed6":            dict(label="SUT fixed-depth",       family="SUT fixed"),
    "sut_common6":           dict(label="SUT",                   family="SUT"),
    "sut_common6_mim_paper": dict(label="SUT, MIM paper sign",   family="SUT"),
    "sut_common6_long":      dict(label="SUT, 3x budget",        family="SUT"),
    "mom_donor6_cls_full":   dict(label="MoM donor (dense)",     family="MoM donor"),
    "mom_ref6_cls_fromfull": dict(label="MoM",                   family="MoM"),
}
# random-initialisation controls: same architecture, untrained weights (probing baseline)
CONTROLS = {"dense6@rand": "dense6", "sut_common6@rand": "sut_common6"}
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
ALL_KEYS = list(MODELS) + list(CONTROLS)
SEED = 0

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d6409f", "#008300", "#6250d6", "#3d3d3a"]
FAMILY_ORDER = ["Dense", "UT", "Switch", "MoA", "SUT", "SUT fixed", "MoM donor", "MoM"]
FAMILY_COLOR = dict(zip(FAMILY_ORDER, PALETTE))
VARIANT_STYLE = {"sut_common6_mim_paper": "--", "sut_common6_long": "-.", "mom_donor6_cls_full": ":"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True,
})

def label_of(key):
    if key in CONTROLS:
        return MODELS[CONTROLS[key]]["label"] + " (random init)"
    return MODELS[key]["label"]

def style(key):
    base = CONTROLS.get(key, key)
    col = FAMILY_COLOR.get(MODELS[base]["family"], INK2)
    return col, (":" if key in CONTROLS else VARIANT_STYLE.get(key, "-"))

def save_json(obj, name):
    (OUT / name).write_text(json.dumps(obj, indent=1))

def load_json(name):
    p = OUT / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run the compute unit that writes it first")
    return json.loads(p.read_text())

missing = [k for k in MODELS if not (RUN_ROOT / f"{k}_s{SEED}" / "best.pt").exists()]
print("ROOT:", ROOT, "| profile:", PROFILE, S)
print("checkpoints missing:", missing or "none")
''')

# ---------------------------------------------------------------------------------------------- 1
md(r"""## Unit 1: Model loading and checkpoint check

`load_model` rebuilds a model from its run's `config.json` and loads `best.pt` (or keeps the random
initialisation for a control). `forward_logits` hides the interface differences: SUT models are encoders in
$T\times B\times C$ layout read out at `<cls>`, MoM and its donor are GPT-2-style blocks in $B\times T\times C$
layout read out at `<cls>` after the final LayerNorm.

The check re-evaluates each model on test examples and compares with the accuracy logged at training time.""")
code(r'''
VOCAB = Vocab()

def load_model(key, seed=SEED):
    base = CONTROLS.get(key, key)
    run_dir = RUN_ROOT / f"{base}_s{seed}"
    cfg = json.loads((run_dir / "config.json").read_text())
    torch.manual_seed(1000 + seed)
    model, interface, _ = build_model(cfg["model"], VOCAB)
    if key not in CONTROLS:
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["model"])
    else:
        # SUT's ParallelExperts initialise their weights uniformly in +-1/fan_out (upstream), i.e. about
        # 1/100 of the usual scale, so at initialisation every block adds almost nothing and the <cls>
        # state is the same for all inputs. A useful control needs random features, so it redraws the
        # expert weights at the usual 1/sqrt(fan_in) scale and keeps the rest of the random initialisation.
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("experts.w"):
                    p.normal_(0, p.shape[-2] ** -0.5)
    model.eval()
    model.kind = cfg["model"]["kind"]
    model.key = key
    return model

def is_sut(model):
    return model.kind.startswith("sut")

def is_halting(model):
    return is_sut(model) and model.encoder.halting

def body(model):
    # the module that owns the per-step loop and the step_hook attribute
    return model.encoder if is_sut(model) else model.transformer

def read_examples(part, n, seed=0):
    exs = logic.read_jsonl(DATA / f"{part}.jsonl")
    rng = random.Random(seed)
    return exs if n >= len(exs) else rng.sample(exs, n)

def encode_batch(exs):
    ids = [logic.encode_classifier(e, VOCAB)[0] for e in exs]
    T = max(len(s) for s in ids)
    x = torch.full((len(ids), T), VOCAB.pad, dtype=torch.long)
    for i, s in enumerate(ids):
        x[i, :len(s)] = torch.tensor(s)
    y = torch.tensor([logic.LABELS.index(e.label) for e in exs])
    return x, x.ne(VOCAB.pad).float(), y

def readout(model, h):
    # h: B x C state of <cls> at the last step -> logits (exactly what the classifier does)
    if is_sut(model):
        if model.encoder.layer_norm is not None:
            h = model.encoder.layer_norm(h)
        return model.head(h)
    return model.head(model.transformer.ln_f(h))

@torch.no_grad()
def forward_logits(model, x, am):
    if is_sut(model):
        return model(x)[0]
    return model(x, attention_mask=am)["logits"]

def batches_by_length(exs, bs=256):
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].p1) + len(exs[i].p2))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        yield idx, [exs[i] for i in idx]

@torch.no_grad()
def accuracy(model, exs, bs=256):
    correct = 0
    for _, b in batches_by_length(exs, bs):
        x, am, y = encode_batch(b)
        correct += (forward_logits(model, x, am).argmax(-1) == y).sum().item()
    return correct / len(exs)

def logged_acc(key, part="test_iid", seed=SEED):
    r = json.loads((RUN_ROOT / f"{key}_s{seed}" / "result.json").read_text())
    return r["test"][part]["acc"] if r.get("test") else r["valid"]["acc"]

def checkpoint_check(force=False):
    path = OUT / "checkpoint_check.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    test_full = logic.read_jsonl(DATA / "test_iid.jsonl")   # all 23,922 test examples (about 10 min on 2 CPU cores)
    rows = []
    for key in MODELS:
        m = load_model(key)
        rows.append(dict(model=key, params=sum(p.numel() for p in m.parameters()),
                         acc_now=accuracy(m, test_full), acc_logged=logged_acc(key)))
    chk = pd.DataFrame(rows); chk["match"] = (chk.acc_now - chk.acc_logged).abs() < 1e-3
    chk.to_csv(path, index=False)
    return chk

if __name__ == "__main__":
    chk = checkpoint_check()
    display(chk.round(4))
    assert chk.match.all(), "a reloaded model does not reproduce its logged test accuracy"
''')

# ---------------------------------------------------------------------------------------------- 2
md(r"""## Unit 2: Semantic targets

For each example we compute what a model *should* represent if it solves the task the way the generator
defines it:

* **label** (7 classes) and the **region bits** $r_1..r_4$ (the label is a function of them);
* each clause's **truth table**: a 64-bit vector $t_A[w] = [w \in S_A]$;
* the clause's **top operator** (variable / not / and / or), a purely syntactic control target.

Validation: rebuilding the label from the region bits must give the stored label for every test example.""")
code(r'''
REGION_TO_LABEL = {(1, 0, 0, 1): "=", (1, 0, 1, 1): "<", (1, 1, 0, 1): ">", (0, 1, 1, 0): "^",
                   (0, 1, 1, 1): "|", (1, 1, 1, 0): "v", (1, 1, 1, 1): "#"}
REGION_NAMES = [r"$r_1$: A∧B", r"$r_2$: A∧¬B", r"$r_3$: ¬A∧B", r"$r_4$: ¬A∧¬B"]
TOP_OPS = ["var", "not", "and", "or"]

def truth_set(tokens):
    return logic.satisfying_worlds(logic.parse(tokens))

def truth_vector(s):
    v = np.zeros(64, dtype=np.uint8); v[list(s)] = 1
    return v

def region_bits(sa, sb):
    U = logic.ALL_WORLDS
    return np.array([bool(sa & sb), bool(sa - sb), bool(sb - sa), bool(U - (sa | sb))], dtype=np.uint8)

def top_op(tokens):
    t = logic.parse(tokens)
    if isinstance(t, str):
        return 0
    return 1 if t[0] == "not" else (2 if t[1] == "and" else 3)

def build_targets(exs):
    out = {k: [] for k in ("label", "region", "ttA", "ttB", "opA", "opB", "len_bin")}
    for e in exs:
        sa, sb = truth_set(e.p1), truth_set(e.p2)
        out["label"].append(logic.LABELS.index(e.label))
        out["region"].append(region_bits(sa, sb))
        out["ttA"].append(truth_vector(sa)); out["ttB"].append(truth_vector(sb))
        out["opA"].append(top_op(e.p1)); out["opB"].append(top_op(e.p2))
        out["len_bin"].append(e.length_bin)
    return {k: np.array(v) for k, v in out.items()}

# token-type ids for routing analysis
TOKEN_TYPES = ["<cls>", "#SEP#", "(", ")", "var", "and", "or", "not"]
def token_type(tok_id):
    s = VOCAB.itos[tok_id]
    if s in logic.VARIABLES:
        return TOKEN_TYPES.index("var")
    return TOKEN_TYPES.index(s) if s in TOKEN_TYPES else -1

if __name__ == "__main__":
    exs = logic.read_jsonl(DATA / "test_iid.jsonl")
    tg = build_targets(exs)
    rebuilt = np.array([logic.LABELS.index(REGION_TO_LABEL[tuple(r)]) for r in tg["region"]])
    print("label rebuilt from region bits matches:", (rebuilt == tg["label"]).mean())
    assert (rebuilt == tg["label"]).all()
    print("region-bit base rates:", dict(zip(["r1", "r2", "r3", "r4"], tg["region"].mean(0).round(3))))
    print("label distribution:", dict(zip(logic.LABELS, np.bincount(tg["label"], minlength=7))))
''')

# ---------------------------------------------------------------------------------------------- 3
md(r"""## Unit 3: Minimal pairs for activation patching

A *corrupt* input differs from the *clean* one in exactly one token of one clause (a variable replaced by
another variable, or `and`↔`or`). The edit keeps the length, so positions align, and it must change the gold
label (and keep both clauses non-degenerate). Patching then asks which states carry the information that
the edit changed.""")
code(r'''
def corrupt_candidates(e):
    for side in (0, 1):
        clause = e.p1 if side == 0 else e.p2
        for pos, tok in enumerate(clause):
            if tok in logic.VARIABLES:
                subs = [v for v in logic.VARIABLES if v != tok]
            elif tok in ("and", "or"):
                subs = ["or" if tok == "and" else "and"]
            else:
                continue
            for s in subs:
                yield side, pos, s

def make_pairs(exs, n, seed=0):
    rng = random.Random(seed)
    exs = list(exs); rng.shuffle(exs)
    pairs = []
    for e in exs:
        cands = list(corrupt_candidates(e)); rng.shuffle(cands)
        for side, pos, s in cands:
            p1, p2 = list(e.p1), list(e.p2)
            (p1 if side == 0 else p2)[pos] = s
            sa, sb = truth_set(p1), truth_set(p2)
            if not sa or not sb or sa == logic.ALL_WORLDS or sb == logic.ALL_WORLDS:
                continue
            lab = logic.compute_relation(sa, sb)
            if lab == e.label:
                continue
            c = Example(label=lab, p1=p1, p2=p2, source=e.source, line=e.line, ops1=e.ops1, ops2=e.ops2)
            edit_index = 1 + pos if side == 0 else 1 + len(e.p1) + 1 + pos   # index in <cls> p1 #SEP# p2
            pairs.append(dict(clean=e, corrupt=c, side=side, edit_index=edit_index,
                              kind="var" if s in logic.VARIABLES else "op"))
            break
        if len(pairs) >= n:
            break
    return pairs

PATCH_GROUPS = ["<cls>", "edited token", "rest of edited clause", "#SEP#", "other clause", "all"]

def group_masks(pair, T):
    e = pair["clean"]; la, lb = len(e.p1), len(e.p2)
    A = list(range(1, 1 + la)); sep = 1 + la; Bp = list(range(sep + 1, sep + 1 + lb))
    edited, other = (A, Bp) if pair["side"] == 0 else (Bp, A)
    m = {g: torch.zeros(T, dtype=torch.bool) for g in PATCH_GROUPS}
    m["<cls>"][0] = True
    m["edited token"][pair["edit_index"]] = True
    m["rest of edited clause"][[i for i in edited if i != pair["edit_index"]]] = True
    m["#SEP#"][sep] = True
    m["other clause"][other] = True
    m["all"][: 2 + la + lb] = True
    return m

if __name__ == "__main__":
    pairs = make_pairs(read_examples("test_iid", 4000, seed=1), 8, seed=1)
    for p in pairs[:6]:
        print(f"{p['clean'].label} -> {p['corrupt'].label} | {' '.join(p['clean'].p1)}  #SEP#  {' '.join(p['clean'].p2)}"
              f"\n      corrupt: {' '.join(p['corrupt'].p1)}  #SEP#  {' '.join(p['corrupt'].p2)}  (edit at {p['edit_index']}, {p['kind']})")
''')

# ---------------------------------------------------------------------------------------------- 4
md(r"""## Unit 4: Capturing hidden states at every step

The models expose a `step_hook` (added for this analysis; it is `None` during training, so training is
unchanged). It is called on the embeddings (step $-1$) and after every block application. For halting SUT
models the representation that the classifier would read if computation stopped at step $i$ is the
soft-halted state

$$\tilde h_t^{(i)} = \sum_{j<i} p_t^{(j)} h_t^{(j)} + \Big(1 - \sum_{j<i} p_t^{(j)}\Big) h_t^{(i)},$$

so that is what we record for them; for every other model it is the block output $h^{(i)}$.

For each step we store the `<cls>` state and the mean over the tokens of clause A and of clause B, plus
per-token routing decisions and (for halting models) whether the token is still active.""")
code(r'''
def _sut_rep(st):
    return st["soft"] if st["soft"] is not None else st["x"]

@torch.no_grad()
def capture(model, exs, bs=256, want_routes=False):
    """Returns dict with cls/poolA/poolB: [steps] x N x C (float16), logits N x 7, and optional routing."""
    N = len(exs)
    store = {"cls": {}, "poolA": {}, "poolB": {}}
    routes = []          # list of per-batch dicts: step -> module -> (B x T) expert ids, plus types / active
    logits_all = torch.zeros(N, 7)
    moes = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "MoE" and m.num_experts > 1]

    for idx, b in batches_by_length(exs, bs):
        x, am, y = encode_batch(b)
        B, T = x.shape
        la = torch.tensor([len(e.p1) for e in b]); lb = torch.tensor([len(e.p2) for e in b])
        pos = torch.arange(T)[None]
        mA = ((pos >= 1) & (pos < 1 + la[:, None])).float()
        mB = ((pos >= 2 + la[:, None]) & (pos < 2 + la[:, None] + lb[:, None])).float()
        rec = {}

        def put(step, H):  # H: B x T x C
            rec[step] = (H[:, 0].clone(), (H * mA[..., None]).sum(1) / mA.sum(1, keepdim=True),
                         (H * mB[..., None]).sum(1) / mB.sum(1, keepdim=True))

        r_batch = {}
        if is_sut(model):
            def hook(i, st):
                put(i, _sut_rep(st).transpose(0, 1))
                if want_routes and i >= 0:
                    d = {}
                    for n, m in moes:
                        if ".layers." in "." + n and f"layers.{i}." not in n:
                            continue   # untied model: only the modules of the layer that just ran
                        top = m.last_top_k_indices[:, 0].view(T, B).t()  # rows of the MoE input are T-major
                        d[n.replace("encoder.", "")] = top.clone()
                    act = (st["p_never_halt"] > 0).t() if st["p_never_halt"] is not None else am.bool()
                    r_batch[i] = (d, act.clone())
                return st
            model.encoder.step_hook = hook
        else:
            def hook(k, h, r):
                put(k, h)
                if want_routes and r is not None:
                    ra, rf = r
                    r_batch[k] = ({"attn router": ra[:, 0].view(B, T).clone(), "ffn router": rf[:, 0].view(B, T).clone()},
                                  am.bool())
                return h
            model.transformer.step_hook = hook
        try:
            logits = forward_logits(model, x, am)
        finally:
            body(model).step_hook = None
        logits_all[idx] = logits.float()
        for step, (c, pa, pb) in rec.items():
            for name, v in (("cls", c), ("poolA", pa), ("poolB", pb)):
                store[name].setdefault(step, torch.zeros(N, v.shape[-1], dtype=torch.float16))[idx] = v.half()
        if want_routes:
            types = torch.tensor([[token_type(t) if t != VOCAB.pad else -1 for t in row] for row in x.tolist()])
            routes.append(dict(types=types, steps=r_batch))
    steps = sorted(store["cls"])
    out = {name: torch.stack([store[name][s] for s in steps]) for name in store}
    out["steps"] = steps
    out["logits"] = logits_all
    if want_routes:
        out["routes"] = routes
    return out

def split_examples():
    return {"train": read_examples("train", S["n_train"], seed=11),
            "test": read_examples("test_iid", S["n_test"], seed=12),
            "long": read_examples("test_len", S["n_long"], seed=13)}

def capture_all(keys=ALL_KEYS, force=False):
    exs = split_examples()
    for key in keys:
        path = OUT / f"states_{key}.pt"
        if path.exists() and not force:
            continue
        t0 = time.time(); m = load_model(key)
        res = {sp: capture(m, e, want_routes=(sp == "test")) for sp, e in exs.items()}
        torch.save(res, path)
        print(f"captured {key:24s} steps={res['test']['steps']} ({time.time() - t0:.0f}s)")
    return exs

def load_states(key):
    p = OUT / f"states_{key}.pt"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run Unit 4")
    return torch.load(p, weights_only=False)

if __name__ == "__main__":
    EXS = capture_all()
    TARGETS = {sp: build_targets(e) for sp, e in EXS.items()}
    # validation: the captured last-step <cls> state must reproduce the model's own logits through the readout
    rows = []
    for key in ALL_KEYS:
        st = load_states(key); m = load_model(key)
        with torch.no_grad():
            re = readout(m, st["test"]["cls"][-1].float())
        rows.append(dict(model=key, steps=len(st["test"]["steps"]),
                         max_logit_err=(re - st["test"]["logits"]).abs().max().item(),
                         acc=(st["test"]["logits"].argmax(-1).numpy() == TARGETS["test"]["label"]).mean()))
    chk = pd.DataFrame(rows); display(chk.round(4))
    assert (chk.max_logit_err < 0.05).all(), "captured readout state does not reproduce the logits"
''')

# ---------------------------------------------------------------------------------------------- 5
md(r"""## Unit 5: Linear probing

A linear probe $\hat y = \operatorname{softmax}(W z + b)$ (or a sigmoid per bit) is trained on states $z$ from
**training** examples and evaluated on held-out **test** examples (lengths 0–6) and on **longer** formulas
(lengths 7–12). Features are standardised with training statistics; the probe has weight decay
$\lambda\lVert W\rVert^2$ and is fit by L-BFGS. We report **balanced accuracy** (the mean of per-class recalls,
and for bit vectors the mean over bits of $\tfrac12(\mathrm{TPR}+\mathrm{TNR})$), so chance is $0.5$ for bits and
$1/7$ for the label irrespective of class imbalance.

Targets (feature → target):

| Probe | Reads | Predicts | What success means |
|---|---|---|---|
| `label` | `<cls>` | 7-way relation | the answer is linearly available |
| `region` | `<cls>` | $r_1..r_4$ | the set-relation facts the label is built from |
| `tt(A)@cls`, `tt(B)@cls` | `<cls>` | 64-bit truth table | `<cls>` holds each clause's full meaning |
| `tt(A)@A`, `tt(B)@B` | mean over the clause's tokens | its truth table | the clause is evaluated locally |
| `topop(A)` | `<cls>` | var/not/and/or | syntactic control |

The **random-init controls** (same architecture, untrained) show how much a probe can read from the input
alone, so the gap between a trained model and its control is the part the model computed.

We also measure the geometry of the `<cls>` states per step: the **participation ratio**
$\mathrm{PR} = (\sum_k \lambda_k)^2 / \sum_k \lambda_k^2$ of their covariance spectrum (effective dimension) and
the mean pairwise cosine similarity (near 1 = all inputs map to the same direction, i.e. collapse).""")
code(r'''
def _standardise(Xtr, *Xs):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-5
    return [(X - mu) / sd for X in (Xtr,) + Xs]

def fit_probe(Xtr, Ytr, Xevals, task, wd=1e-3, iters=150):
    """task: 'multiclass' (Y: N) or 'multilabel' (Y: N x K). Returns balanced accuracy on each eval set."""
    Xtr, *Xevals = _standardise(Xtr.float(), *[X.float() for X in Xevals])
    K = int(Ytr.max()) + 1 if task == "multiclass" else Ytr.shape[1]
    lin = nn.Linear(Xtr.shape[1], K)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    Yt = torch.as_tensor(Ytr).long() if task == "multiclass" else torch.as_tensor(Ytr).float()

    def closure():
        opt.zero_grad()
        out = lin(Xtr)
        loss = F.cross_entropy(out, Yt) if task == "multiclass" else F.binary_cross_entropy_with_logits(out, Yt)
        loss = loss + wd * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return [lin(X) for X in Xevals]

def balanced_acc(out, Y, task):
    Y = np.asarray(Y)
    if task == "multiclass":
        pred = out.argmax(-1).numpy()
        recalls = [(pred[Y == c] == c).mean() for c in np.unique(Y)]
        return float(np.mean(recalls))
    pred = (out > 0).numpy().astype(np.uint8)
    scores = []
    for k in range(Y.shape[1]):
        y = Y[:, k]
        if y.min() == y.max():
            continue
        scores.append(0.5 * ((pred[y == 1, k] == 1).mean() + (pred[y == 0, k] == 0).mean()))
    return float(np.mean(scores))

PROBES = [  # name, feature, target, task
    ("label", "cls", "label", "multiclass"),
    ("region", "cls", "region", "multilabel"),
    ("tt(A)@cls", "cls", "ttA", "multilabel"),
    ("tt(B)@cls", "cls", "ttB", "multilabel"),
    ("tt(A)@A", "poolA", "ttA", "multilabel"),
    ("tt(B)@B", "poolB", "ttB", "multilabel"),
    ("topop(A)", "cls", "opA", "multiclass"),
]

def geometry(Z):
    Z = Z.float()
    Zc = Z - Z.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(Zc.T @ Zc / len(Z)).clamp(min=0)
    pr = (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12)).item()
    Zn = F.normalize(Z, dim=-1); sub = Zn[:1000]
    cos = ((sub @ sub.T).sum() - len(sub)) / (len(sub) * (len(sub) - 1))
    return pr, cos.item()

def run_probes(keys=ALL_KEYS, force=False):
    path = OUT / "probes.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    exs = split_examples()
    tg = {sp: build_targets(e) for sp, e in exs.items()}
    rows = []
    for key in keys:
        st = load_states(key); t0 = time.time()
        for si, step in enumerate(st["test"]["steps"]):
            for name, feat, target, task in PROBES:
                outs = fit_probe(st["train"][feat][si], tg["train"][target],
                                 [st["test"][feat][si], st["long"][feat][si]], task)
                for sp, o in zip(("test", "long"), outs):
                    rows.append(dict(model=key, step=step, probe=name, split=sp,
                                     bacc=balanced_acc(o, tg[sp][target], task)))
            pr, cos = geometry(st["test"]["cls"][si])
            rows.append(dict(model=key, step=step, probe="PR", split="test", bacc=pr))
            rows.append(dict(model=key, step=step, probe="cos", split="test", bacc=cos))
        print(f"probed {key:24s} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows); df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    PR_DF = run_probes()
    last = PR_DF[(PR_DF.split == "test")].groupby(["model", "probe"]).apply(
        lambda d: d.sort_values("step").bacc.iloc[-1]).unstack()
    display(last[[p[0] for p in PROBES] + ["PR", "cos"]].round(3))
''')

md("### Unit 5a: Figure: what `<cls>` encodes at each step (label and region bits, test and long)")
code(r'''
def fig_probe_curves(df, fname="a5a_probe_label_region.png"):
    keys = [k for k in ALL_KEYS if k in set(df.model)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    for col, probe, chance in ((0, "label", 1 / 7), (1, "region", 0.5)):
        for row, sp, title in ((0, "test", "held-out test, lengths 0–6"), (1, "long", "longer formulas, lengths 7–12")):
            ax = axes[row, col]
            for k in keys:
                d = df[(df.model == k) & (df.probe == probe) & (df.split == sp)].sort_values("step")
                c, ls = style(k)
                ax.plot(d.step, d.bacc, color=c, ls=ls, lw=1.4 if k in CONTROLS else 2,
                        alpha=0.7 if k in CONTROLS else 1, label=label_of(k), marker="o", ms=3)
            ax.axhline(chance, color=INK2, ls=":", lw=1)
            ax.set_title(f"probe: {probe} from <cls> — {title}", loc="left")
            ax.set_ylabel("balanced accuracy")
    for ax in axes[1]:
        ax.set_xlabel("step (−1 = embeddings; one step = one block application)")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_probe_curves(run_probes())
''')

md("### Unit 5b: Figure: clause meaning (truth tables) at the clause tokens vs at `<cls>`")
code(r'''
def fig_truth_tables(df, fname="a5b_probe_truth_tables.png"):
    keys = [k for k in ALL_KEYS if k in set(df.model)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    panels = [("tt(A)@A", "clause A tokens → truth table of A"), ("tt(A)@cls", "<cls> → truth table of A"),
              ("tt(B)@cls", "<cls> → truth table of B")]
    for ax, (probe, title) in zip(axes, panels):
        for k in keys:
            d = df[(df.model == k) & (df.probe == probe) & (df.split == "test")].sort_values("step")
            c, ls = style(k)
            ax.plot(d.step, d.bacc, color=c, ls=ls, label=label_of(k), marker="o", ms=3,
                    lw=1.4 if k in CONTROLS else 2, alpha=0.7 if k in CONTROLS else 1)
        ax.axhline(0.5, color=INK2, ls=":", lw=1)
        ax.set_title(title, loc="left"); ax.set_xlabel("step")
    axes[0].set_ylabel("mean balanced accuracy over 64 worlds")
    axes[0].legend(fontsize=7, ncol=1, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_truth_tables(run_probes())
''')

md("### Unit 5c: Figure: geometry of the `<cls>` states (effective dimension and collapse)")
code(r'''
def fig_geometry(df, fname="a5c_cls_geometry.png"):
    keys = [k for k in ALL_KEYS if k in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, probe, title in ((axes[0], "PR", "participation ratio of <cls> states (effective dimension)"),
                             (axes[1], "cos", "mean pairwise cosine of <cls> states (1 = collapsed)")):
        for k in keys:
            d = df[(df.model == k) & (df.probe == probe)].sort_values("step")
            c, ls = style(k)
            ax.plot(d.step, d.bacc, color=c, ls=ls, label=label_of(k), marker="o", ms=3,
                    lw=1.4 if k in CONTROLS else 2, alpha=0.7 if k in CONTROLS else 1)
        ax.set_title(title, loc="left"); ax.set_xlabel("step")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_geometry(run_probes())
''')

# ---------------------------------------------------------------------------------------------- 6
md(r"""## Unit 6: Activation patching

For a clean/corrupt pair with gold labels $y$ and $y'$ we use the logit difference
$\Delta(\cdot) = \ell_y(\cdot) - \ell_{y'}(\cdot)$. We run the corrupt input, and at step $i$ overwrite the states
of one position group with their values from the clean run, then let the model continue. The **recovered
fraction**

$$R(i, g) = \frac{\sum_{\text{pairs}} \big[\Delta(\text{patched}_{i,g}) - \Delta(\text{corrupt})\big]}
{\sum_{\text{pairs}} \big[\Delta(\text{clean}) - \Delta(\text{corrupt})\big]}$$

is 1 when the patched states carry all the information the edit removed, and 0 when they carry none.
Pairs are kept when the model gets the clean input right with a margin ($\Delta(\text{clean}) - \Delta(\text{corrupt}) > 1$).

For halting SUT models we patch the hidden state, the soft-halted state and the whole ACT state
(accumulated output, survival probability, expected depth), so the patch is exact. Patching the group "all"
must give $R = 1$ at every step: that is the unit's validation. (For MoM the GRU router state carried across
assembly steps is not patched, so there "all" is within 1% of 1.)""")
code(r'''
@torch.no_grad()
def _run_collect(model, x, am):
    """Run and return per-step full states (for patching)."""
    saved = {}
    if is_sut(model):
        def hook(i, st):
            act = st["act"]
            saved[i] = dict(x=st["x"].clone(), soft=None if st["soft"] is None else st["soft"].clone(),
                            act=None if act is None else (act[0],) + tuple(t.clone() for t in act[1:]))
            return st
        model.encoder.step_hook = hook
    else:
        def hook(k, h, r):
            saved[k] = h.clone(); return h
        model.transformer.step_hook = hook
    try:
        logits = forward_logits(model, x, am)
    finally:
        body(model).step_hook = None
    return logits, saved

@torch.no_grad()
def _run_patched(model, x, am, step, mask, clean):
    """mask: B x T bool positions to overwrite with the clean run's state at `step`."""
    if is_sut(model):
        mT = mask.t()  # T x B
        def hook(i, st):
            if i != step:
                return st
            c = clean[i]
            st["x"] = torch.where(mT[..., None], c["x"], st["x"])
            if st["soft"] is not None:
                st["soft"] = torch.where(mT[..., None], c["soft"], st["soft"])
            if st["act"] is not None:
                a, ca = st["act"], c["act"]
                st["act"] = (a[0], torch.where(mT, ca[1], a[1]), torch.where(mT[..., None], ca[2], a[2]),
                             torch.where(mT, ca[3], a[3]))
            return st
        model.encoder.step_hook = hook
    else:
        def hook(k, h, r):
            return torch.where(mask[..., None], clean[k], h) if k == step else h
        model.transformer.step_hook = hook
    try:
        return forward_logits(model, x, am)
    finally:
        body(model).step_hook = None

def run_patching(keys=list(MODELS), force=False):
    path = OUT / "patching.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    pairs = make_pairs(read_examples("test_iid", 20000, seed=21), S["n_pairs"], seed=21)
    by_len = {}
    for p in pairs:
        by_len.setdefault(len(p["clean"].p1) + len(p["clean"].p2), []).append(p)
    rows = []
    for key in keys:
        m = load_model(key); t0 = time.time()
        num = {}; den = 0.0; kept = 0
        for L, ps in sorted(by_len.items()):
            for k in range(0, len(ps), 128):
                b = ps[k:k + 128]
                xc, am, yc = encode_batch([p["clean"] for p in b])
                xr, _, yr = encode_batch([p["corrupt"] for p in b])
                lc, clean = _run_collect(m, xc, am)
                lr = forward_logits(m, xr, am)
                ar = torch.arange(len(b))
                dc = lc[ar, yc] - lc[ar, yr]; dr = lr[ar, yc] - lr[ar, yr]
                keep = (lc.argmax(-1) == yc) & ((dc - dr) > 1.0)
                if keep.sum() == 0:
                    continue
                kept += int(keep.sum()); den += float((dc - dr)[keep].sum())
                T = xc.shape[1]
                gm = [group_masks(p, T) for p in b]
                for step in sorted(clean):
                    for g in PATCH_GROUPS:
                        mask = torch.stack([d[g] for d in gm])
                        lp = _run_patched(m, xr, am, step, mask, clean)
                        dp = lp[ar, yc] - lp[ar, yr]
                        num[(step, g)] = num.get((step, g), 0.0) + float((dp - dr)[keep].sum())
        for (step, g), v in num.items():
            rows.append(dict(model=key, step=step, group=g, recovery=v / den if den else float("nan"), n_pairs=kept))
        print(f"patched {key:24s} pairs kept {kept:4d} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows); df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    PATCH = run_patching()
    allg = PATCH[PATCH.group == "all"]
    print("validation: recovery for group 'all' (must be 1):", allg.recovery.round(4).unique())
    # exact (1.000) for every SUT-family model; MoM deviates by <1% because its GRU router state, which is
    # carried across the assembly steps of the MoTBlock, is not part of the patched hidden state
    assert np.allclose(allg[~allg.model.str.startswith("mom_ref")].recovery, 1.0, atol=1e-3)
    assert np.allclose(allg.recovery.dropna(), 1.0, atol=2e-2)
    display(PATCH.groupby("model").n_pairs.first())
''')

md("### Unit 6a: Figure: patching heatmaps (step × position group)")
code(r'''
def fig_patching(df, keys=None, fname="a6a_patching.png"):
    keys = keys or [k for k in MODELS if k in set(df.model) and df[df.model == k].n_pairs.iloc[0] >= 20]
    groups = [g for g in PATCH_GROUPS if g != "all"]
    n = len(keys); cols = 4; rows_ = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(3.4 * cols + 1.2, 3.1 * rows_), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.08)
    for i, (ax, k) in enumerate(zip(axes.flat, keys)):
        d = df[df.model == k].pivot_table(index="group", columns="step", values="recovery").reindex(groups)
        im = ax.imshow(d.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(d.shape[1]), d.columns)
        ax.set_yticks(range(len(groups)), groups if i % cols == 0 else [""] * len(groups), fontsize=7.5)
        ax.grid(False)
        for (r, c), v in np.ndenumerate(d.values):
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else INK)
        ax.set_title(f"{label_of(k)} (n={int(df[df.model == k].n_pairs.iloc[0])})", loc="left", fontsize=9)
        ax.set_xlabel("patched step")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="recovered fraction R")
    fig.savefig(FIG / fname, bbox_inches="tight"); plt.show()

fig_patching(run_patching())
''')

md("### Unit 6b: Figure: when does the answer reach `<cls>`?")
code(r'''
def fig_patching_cls(df, fname="a6b_patching_cls_vs_edit.png"):
    keys = [k for k in MODELS if k in set(df.model) and df[df.model == k].n_pairs.iloc[0] >= 20]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, g, title in ((axes[0], "<cls>", "patch <cls> only"), (axes[1], "edited token", "patch the edited token only")):
        for k in keys:
            d = df[(df.model == k) & (df.group == g)].sort_values("step")
            c, ls = style(k)
            ax.plot(d.step, d.recovery, color=c, ls=ls, marker="o", ms=3, label=label_of(k))
        ax.set_title(title, loc="left"); ax.set_xlabel("patched step"); ax.axhline(0, color=INK2, lw=0.8)
    axes[0].set_ylabel("recovered fraction R"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_patching_cls(run_patching())
''')

# ---------------------------------------------------------------------------------------------- 7
md(r"""## Unit 7: Sparse autoencoder on the `<cls>` readout state

A ReLU sparse autoencoder with $D = 4C = 512$ features decomposes the final `<cls>` state $z\in\mathbb R^{C}$:

$$f = \mathrm{ReLU}\big(W_e (z - b_d) + b_e\big), \qquad \hat z = W_d f + b_d,$$
$$\mathcal L = \lVert z - \hat z\rVert_2^2 + \lambda \sum_j f_j \lVert W_{d,:j}\rVert_2 .$$

Inputs are scaled so that $\mathbb E\lVert z\rVert^2 = C$; $\lambda$ is chosen from the grid $\{0.5, 1, 2\}$ so that
about 15 features are active per input ($L_0$); smaller $\lambda$ gives dense, uninterpretable codes ($L_0 > 100$). We report the fraction of variance explained, $L_0$, dead features, the
accuracy when the model reads the **reconstruction** instead of $z$ ("spliced accuracy"), and how well single
features align with logical concepts: for each feature the largest $|\mathrm{corr}|$ with any concept (label
one-hot, region bits, top operator of A or B). A feature with $|r| > 0.5$ is counted as *concept-aligned*.""")
code(r'''
class SAE(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.We = nn.Parameter(torch.randn(D, C) * (1 / math.sqrt(C)))
        self.Wd = nn.Parameter(self.We.data.T.clone())
        self.be = nn.Parameter(torch.zeros(D)); self.bd = nn.Parameter(torch.zeros(C))

    def encode(self, z):
        return F.relu((z - self.bd) @ self.We.T + self.be)

    def forward(self, z):
        f = self.encode(z)
        return f @ self.Wd.T + self.bd, f

def train_sae(Z, D=512, l1=1.0, steps=4000, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    sae = SAE(Z.shape[1], D); opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for t in range(steps):
        z = Z[torch.randint(len(Z), (bs,))]
        zh, f = sae(z)
        loss = (z - zh).pow(2).sum(-1).mean() + l1 * (f * sae.Wd.norm(dim=0)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return sae

def concept_matrix(tg):
    cols, names = [], []
    for c in range(7):
        cols.append(tg["label"] == c); names.append(f"label {logic.LABELS[c]}")
    for k in range(4):
        cols.append(tg["region"][:, k] == 1); names.append(f"r{k + 1}")
    for side in ("A", "B"):
        for o, nm in enumerate(TOP_OPS):
            cols.append(tg[f"op{side}"] == o); names.append(f"top({side})={nm}")
    return np.stack(cols, 1).astype(np.float32), names

def corr_matrix(Fm, Cm):
    Fz = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-8); Cz = (Cm - Cm.mean(0)) / (Cm.std(0) + 1e-8)
    return Fz.T @ Cz / len(Fm)   # D x K

SAE_KEYS = ["dense6", "switch6", "sut_fixed6", "sut_common6", "sut_common6_long", "mom_ref6_cls_fromfull"]

def run_sae(keys=SAE_KEYS, force=False, l1_grid=(0.5, 1.0, 2.0), target_l0=15):
    path = OUT / "sae.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    exs = split_examples(); tg = build_targets(exs["test"])
    Cm, cnames = concept_matrix(tg)
    res = {}
    for key in keys:
        st = load_states(key); m = load_model(key); t0 = time.time()
        Ztr, Zte = st["train"]["cls"][-1].float(), st["test"]["cls"][-1].float()
        scale = Ztr.pow(2).sum(-1).mean().div(Ztr.shape[1]).sqrt()
        best = None
        for l1 in l1_grid:
            sae = train_sae(Ztr / scale, l1=l1, steps=S["sae_steps"])
            with torch.no_grad():
                zh, f = sae(Zte / scale)
            l0 = (f > 0).float().sum(-1).mean().item()
            if best is None or abs(l0 - target_l0) < abs(best[2] - target_l0):
                best = (l1, sae, l0, zh, f)
        l1, sae, l0, zh, f = best
        with torch.no_grad():
            fve = 1 - ((Zte / scale - zh).pow(2).sum() / (Zte / scale - (Zte / scale).mean(0)).pow(2).sum()).item()
            acc_orig = (readout(m, Zte).argmax(-1).numpy() == tg["label"]).mean()
            acc_spliced = (readout(m, zh * scale).argmax(-1).numpy() == tg["label"]).mean()
        Fm = f.numpy(); live = Fm.max(0) > 0
        R = np.nan_to_num(corr_matrix(Fm[:, live], Cm))
        best_r = np.abs(R).max(1)
        top = np.argsort(-best_r)[:12]
        res[key] = dict(l1=l1, L0=l0, fve=fve, dead=float(1 - live.mean()), acc=float(acc_orig),
                        acc_spliced=float(acc_spliced), aligned=float((best_r > 0.5).mean()),
                        concept_cover=dict(zip(cnames, np.abs(R).max(0).round(3).tolist())),
                        top_corr=R[top].round(3).tolist(), concepts=cnames)
        print(f"SAE {key:24s} L0={l0:5.1f} FVE={fve:.3f} dead={res[key]['dead']:.2f} "
              f"acc {acc_orig:.3f}->{acc_spliced:.3f} aligned={res[key]['aligned']:.2f} ({time.time() - t0:.0f}s)")
    path.write_text(json.dumps(res, indent=1))
    return res

if __name__ == "__main__":
    SAE_RES = run_sae()
    display(pd.DataFrame({k: {c: v[c] for c in ("L0", "fve", "dead", "acc", "acc_spliced", "aligned")}
                          for k, v in SAE_RES.items()}).T.round(3))
''')

md("### Unit 7a: Figure: SAE quality and concept alignment")
code(r'''
def fig_sae_summary(res, fname="a7a_sae_summary.png"):
    keys = list(res)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    y = np.arange(len(keys))
    cols = [style(k)[0] for k in keys]
    axes[0].barh(y, [res[k]["fve"] for k in keys], color=cols)
    axes[0].set_title("variance explained by the SAE", loc="left")
    axes[1].barh(y - 0.2, [res[k]["acc"] for k in keys], height=0.38, color=INK2, label="original <cls>")
    axes[1].barh(y + 0.2, [res[k]["acc_spliced"] for k in keys], height=0.38, color=cols, label="SAE reconstruction")
    axes[1].set_xlim(0.4, 0.9); axes[1].legend(fontsize=7); axes[1].set_title("accuracy read from", loc="left")
    axes[2].barh(y, [res[k]["aligned"] for k in keys], color=cols)
    axes[2].set_title("live features with |r| > 0.5 to a concept", loc="left")
    for ax in axes:
        ax.set_yticks(y, [label_of(k) for k in keys] if ax is axes[0] else [""] * len(keys)); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_sae_summary(run_sae())
''')

md("### Unit 7b: Figure: which concepts the best single SAE features capture")
code(r'''
def fig_sae_concepts(res, keys=("dense6", "sut_common6", "sut_common6_long", "mom_ref6_cls_fromfull"),
                     fname="a7b_sae_concepts.png"):
    keys = [k for k in keys if k in res]
    cn = res[keys[0]]["concepts"]
    M = np.array([[res[k]["concept_cover"][c] for c in cn] for k in keys])
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(keys) + 1.6))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues", aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(cn)), cn, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(keys)), [label_of(k) for k in keys])
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else INK)
    ax.set_title("best |correlation| between any single SAE feature and each concept", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.savefig(FIG / fname, bbox_inches="tight"); plt.show()

fig_sae_concepts(run_sae())
''')

# ---------------------------------------------------------------------------------------------- 8
md(r"""## Unit 8: Routing analysis

From the routing decisions recorded in Unit 4 (top-1 expert per token, per step, per routed module; only
tokens that are still active), we compute for every module and step

* the normalised expert entropy $H(E)/\log|E|$ (1 = uniform use, 0 = one expert takes everything), and
* the normalised mutual information between expert and token type, $I(E;\text{type})/H(E)$
  (how much of the routing is explained by *which kind of token* it is: variable, operator, bracket, …).

High $I(E;\text{type})$ with low step dependence means the router acts as a token-type lookup table rather than
a computation-dependent choice.""")
code(r'''
def _entropy(p):
    p = p[p > 0]; return float(-(p * np.log(p)).sum())

def routing_stats(keys=None, force=False):
    path = OUT / "routing.csv"
    if path.exists() and not force:
        return pd.read_csv(path), json.loads((OUT / "routing_tables.json").read_text())
    keys = keys or [k for k in MODELS if k not in ("dense6", "ut_fixed6", "mom_donor6_cls_full")]
    rows, tables = [], {}
    for key in keys:
        st = load_states(key)["test"]
        m = load_model(key)
        if is_sut(m):   # true number of experts per routed module (not just the ones that were used)
            n_exp = {n.replace("encoder.", ""): mm.num_experts for n, mm in m.named_modules() if type(mm).__name__ == "MoE"}
        else:           # MoM: N modules + SKIP
            n_exp = {"attn router": m.config.num_attn + 1, "ffn router": m.config.num_ffn + 1}
        counts = {}
        for b in st["routes"]:
            types = b["types"]
            for step, (mods, act) in b["steps"].items():
                ok = act & (types >= 0)
                for mod, ids in mods.items():
                    t = types[ok].numpy(); e = ids[ok].numpy()
                    c = counts.setdefault((mod, step), np.zeros((len(TOKEN_TYPES), 8)))
                    np.add.at(c, (t, e), 1)
        for (mod, step), c in counts.items():
            c = c[:, :n_exp[mod]]
            pe = c.sum(0) / c.sum(); pt = c.sum(1) / c.sum(); pj = c / c.sum()
            He, Ht, Hj = _entropy(pe), _entropy(pt), _entropy(pj.ravel())
            mi = He + Ht - Hj
            rows.append(dict(model=key, module=mod, step=step, n_tokens=int(c.sum()), n_experts=c.shape[1],
                             H_norm=He / np.log(c.shape[1]), MI_norm=mi / He if He > 1e-9 else 0.0))
            tables[f"{key}|{mod}|{step}"] = (c / np.maximum(c.sum(1, keepdims=True), 1)).round(4).tolist()
    df = pd.DataFrame(rows); df.to_csv(path, index=False)
    (OUT / "routing_tables.json").write_text(json.dumps(tables))
    return df, tables

if __name__ == "__main__":
    RT, RT_TABLES = routing_stats()
    display(RT.groupby(["model", "module"])[["H_norm", "MI_norm"]].mean().round(3))
''')

md("### Unit 8a: Figure: routing entropy and token-type dependence per step")
code(r'''
def fig_routing(df, fname="a8a_routing_entropy_mi.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in df.model.unique():
        c, ls = style(key)
        d = df[df.model == key].groupby("step")[["H_norm", "MI_norm"]].mean().reset_index()
        axes[0].plot(d.step, d.H_norm, color=c, ls=ls, marker="o", ms=3, label=label_of(key))
        axes[1].plot(d.step, d.MI_norm, color=c, ls=ls, marker="o", ms=3, label=label_of(key))
    axes[0].set_title("expert entropy H(E)/log|E| (mean over routed modules)", loc="left")
    axes[1].set_title("token-type share of routing  I(E;type)/H(E)", loc="left")
    for ax in axes:
        ax.set_xlabel("step"); ax.set_ylim(0, 1.05)
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_routing(routing_stats()[0])
''')

md("### Unit 8b: Figure: SUT expert choice by token type over the recurrent steps")
code(r'''
def fig_routing_tables(tables, key="sut_common6", fname="a8b_sut_routing_by_type.png"):
    mods = sorted({k.split("|")[1] for k in tables if k.startswith(key + "|")})
    steps = sorted({int(k.split("|")[2]) for k in tables if k.startswith(key + "|")})[:4]
    fig, axes = plt.subplots(len(mods), len(steps), figsize=(3.0 * len(steps), 2.6 * len(mods)), squeeze=False)
    for r, mod in enumerate(mods):
        for c, step in enumerate(steps):
            ax = axes[r, c]; M = np.array(tables[f"{key}|{mod}|{step}"])
            ax.imshow(M, vmin=0, vmax=1, cmap="Blues", aspect="auto"); ax.grid(False)
            ax.set_xticks(range(M.shape[1]), [f"e{j}" for j in range(M.shape[1])], fontsize=7)
            ax.set_yticks(range(len(TOKEN_TYPES)), TOKEN_TYPES if c == 0 else [""] * len(TOKEN_TYPES), fontsize=7)
            for (i, j), v in np.ndenumerate(M):
                if v > 0.05:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else INK)
            ax.set_title(f"{mod.split('.')[-1] if '.' in mod else mod}, step {step}", fontsize=8.5, loc="left")
    fig.suptitle(f"{label_of(key)}: P(top-1 expert | token type)", x=0.01, ha="left", fontsize=10.5, weight="bold")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_routing_tables(routing_stats()[1])
''')

# ---------------------------------------------------------------------------------------------- 9
md(r"""## Unit 9: Ablation: depth truncation

Run only the first $k$ block applications and read the classifier from the state after step $k$ (for halting
models this is the soft-halted state, i.e. forced early exit). A model that really uses its depth loses
accuracy when truncated; a model whose accuracy is flat after step 2 effectively is a 2-layer network.""")
code(r'''
def run_depth_truncation(keys=None, force=False):
    path = OUT / "ablation_depth.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    keys = keys or [k for k in MODELS if k.startswith(("dense", "ut", "switch", "moa", "sut"))]
    exs = {"test": read_examples("test_iid", S["n_ablate"], seed=31), "long": read_examples("test_len", S["n_ablate"], seed=32)}
    rows = []
    for key in keys:
        m = load_model(key)
        for k in range(1, m.encoder.num_layers + 1):
            m.encoder.run_steps = k
            for sp, e in exs.items():
                rows.append(dict(model=key, steps=k, split=sp, acc=accuracy(m, e)))
        m.encoder.run_steps = None
    df = pd.DataFrame(rows); df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    DEPTH = run_depth_truncation()
    display(DEPTH[DEPTH.split == "test"].pivot_table(index="model", columns="steps", values="acc").round(3))
''')

md("### Unit 9a: Figure: accuracy vs number of executed steps")
code(r'''
def fig_depth(df, fname="a9a_depth_truncation.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, sp, title in ((axes[0], "test", "test, lengths 0–6"), (axes[1], "long", "lengths 7–12")):
        for key in df.model.unique():
            d = df[(df.model == key) & (df.split == sp)]
            c, ls = style(key)
            ax.plot(d.steps, d.acc, color=c, ls=ls, marker="o", ms=3, label=label_of(key))
        ax.set_title(f"depth truncation — {title}", loc="left"); ax.set_xlabel("block applications executed")
    axes[0].set_ylabel("accuracy"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_depth(run_depth_truncation())
''')

# ---------------------------------------------------------------------------------------------- 10
md(r"""## Unit 10: Ablation: halting threshold (SUT's compute–accuracy trade-off)

SUT stops updating a token once its survival probability $\prod_j (1-\alpha_t^{(j)})$ drops below $1-\tau$
(released $\tau = 0.999$). The SUT paper varies this threshold after training to trade compute for
quality. We sweep $\tau$ and record accuracy, the mean expected halting step
$\bar d = \mathbb E_t\big[\sum_i i\,p_t^{(i)}\big]$, and the fraction of token-steps that still run the block.""")
code(r'''
THRESHOLDS = [0.5, 0.8, 0.9, 0.99, 0.999, 0.9999, 1.0]

@torch.no_grad()
def eval_halting(m, exs):
    correct = 0; depth_sum = 0.0; tok = 0; active = 0.0; possible = 0.0
    for _, b in batches_by_length(exs):
        x, am, y = encode_batch(b)
        logits, extra = m(x)
        correct += (logits.argmax(-1) == y).sum().item()
        d = extra["expected_depth_tokens"]          # T x B
        mask = am.t()
        depth_sum += float((d * mask).sum()); tok += float(mask.sum())
        act = extra["encoder_active_tokens"]        # steps
        active += float(act.sum()); possible += float(mask.sum()) * len(act)
    return correct / len(exs), depth_sum / tok, active / possible

def run_threshold_sweep(keys=("sut_common6", "sut_common6_mim_paper", "sut_common6_long"), force=False):
    path = OUT / "ablation_threshold.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    exs = {"test": read_examples("test_iid", S["n_ablate"], seed=31), "long": read_examples("test_len", S["n_ablate"], seed=32)}
    rows = []
    for key in keys:
        m = load_model(key); hl = m.encoder.halting_universal_layer; tau0 = hl.threshold
        for tau in THRESHOLDS:
            hl.threshold = tau
            for sp, e in exs.items():
                acc, d, frac = eval_halting(m, e)
                rows.append(dict(model=key, tau=tau, split=sp, acc=acc, mean_depth=d, active_frac=frac))
        hl.threshold = tau0
    df = pd.DataFrame(rows); df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    THR = run_threshold_sweep()
    display(THR[THR.split == "test"].round(4))
''')

md("### Unit 10a: Figure: accuracy vs compute under the halting-threshold sweep")
code(r'''
def fig_threshold(df, depth_df=None, fname="a10a_halting_threshold.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for key in df.model.unique():
        d = df[(df.model == key) & (df.split == "test")]
        c, ls = style(key)
        axes[0].plot(d.active_frac, d.acc, color=c, ls=ls, marker="o", ms=4, label=label_of(key))
        for _, r in d.iterrows():
            if r.tau in (0.5, 0.999):
                axes[0].annotate(f"τ={r.tau:g}", (r.active_frac, r.acc), xytext=(4, -9), textcoords="offset points", fontsize=6.5, color=INK2)
        axes[1].plot(d.tau.astype(str), d.mean_depth, color=c, ls=ls, marker="o", ms=4, label=label_of(key))
    if depth_df is not None:
        for key in ("dense6", "ut_fixed6"):
            v = depth_df[(depth_df.model == key) & (depth_df.split == "test")].acc.max()
            axes[0].axhline(v, color=style(key)[0], ls=":", lw=1.2)
            axes[0].annotate(label_of(key) + " (all 6 steps)", (1, v), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                             textcoords="offset points", ha="right", fontsize=7, color=style(key)[0])
    axes[0].set_xlabel("fraction of token-steps that run the block"); axes[0].set_ylabel("accuracy (test)")
    axes[0].set_title("accuracy vs compute", loc="left")
    axes[1].set_xlabel("threshold τ"); axes[1].set_ylabel("mean expected halting step")
    axes[1].set_title("how deep tokens go", loc="left"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_threshold(run_threshold_sweep(), run_depth_truncation())
''')

# ---------------------------------------------------------------------------------------------- 11
md(r"""## Unit 11: Ablation: expert knockout and random routing (SUT / Switch / MoA)

**Knockout:** expert $j$ of one routed module gets gate logit $-10^4$, so it is never selected and the top-$k$
gate renormalises over the others. **Random routing:** all gate logits are replaced with Gaussian noise, which
keeps the experts but removes the learned assignment. We report the accuracy change and the mean
$\mathrm{KL}(p_\text{orig}\,\Vert\,p_\text{ablated})$ of the output distribution (sensitive even when accuracy does
not move).""")
code(r'''
class GateEdit:
    """Context manager: edit gate logits of routed modules via forward hooks on MoE.w_gate."""
    def __init__(self, model, fn, modules=None):
        self.model, self.fn = model, fn
        self.targets = [(n, m) for n, m in model.named_modules()
                        if type(m).__name__ == "MoE" and m.num_experts > 1 and (modules is None or n in modules)]
        self.handles = []

    def __enter__(self):
        for n, m in self.targets:
            self.handles.append(m.w_gate.register_forward_hook(lambda mod, inp, out, n=n: self.fn(n, out)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

def knockout_fn(j):
    def f(name, out):
        out = out.clone(); out[:, j] = -1e4; return out
    return f

def random_fn(gen):
    return lambda name, out: torch.randn(out.shape, generator=gen)

@torch.no_grad()
def probs_on(model, exs):
    P = torch.zeros(len(exs), 7); Y = torch.zeros(len(exs), dtype=torch.long)
    for idx, b in batches_by_length(exs):
        x, am, y = encode_batch(b)
        P[idx] = F.softmax(forward_logits(model, x, am).float(), -1); Y[idx] = y
    return P, Y

def kl(P, Q):
    return float((P * (P.clamp_min(1e-9).log() - Q.clamp_min(1e-9).log())).sum(-1).mean())

def run_expert_ablation(keys=("switch6", "moa6", "sut_fixed6", "sut_common6", "sut_common6_long"), force=False):
    path = OUT / "ablation_experts.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    exs = read_examples("test_iid", S["n_ablate"], seed=31)
    rows = []
    for key in keys:
        m = load_model(key); P0, Y = probs_on(m, exs); acc0 = (P0.argmax(-1) == Y).float().mean().item()
        mods = [(n, mm) for n, mm in m.named_modules() if type(mm).__name__ == "MoE" and mm.num_experts > 1]
        for n, mm in mods:
            for j in range(mm.num_experts):
                with GateEdit(m, knockout_fn(j), modules={n}):
                    P, _ = probs_on(m, exs)
                rows.append(dict(model=key, ablation="knockout", module=n.replace("encoder.", ""), expert=j,
                                 acc0=acc0, acc=(P.argmax(-1) == Y).float().mean().item(), kl=kl(P0, P)))
        for r in range(3):
            g = torch.Generator().manual_seed(r)
            with GateEdit(m, random_fn(g)):
                P, _ = probs_on(m, exs)
            rows.append(dict(model=key, ablation="random routing", module="all", expert=-1, acc0=acc0,
                             acc=(P.argmax(-1) == Y).float().mean().item(), kl=kl(P0, P)))
        print("ablated", key)
    df = pd.DataFrame(rows); df["dacc"] = df.acc - df.acc0; df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    EXP = run_expert_ablation()
    display(EXP.groupby(["model", "ablation"])[["dacc", "kl"]].agg(["mean", "min"]).round(4))
''')

md("### Unit 11a: Figure: expert knockout heatmaps and random routing")
code(r'''
def fig_expert_ablation(df, fname="a11a_expert_knockout.png"):
    keys = list(dict.fromkeys(df.model))
    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(2, len(keys), height_ratios=[3, 1.4], hspace=0.45, wspace=0.45)
    def short(s):
        s = s.replace("universal_layer.", "").replace("self_attn.q_proj", "attn").replace("moe", "ffn")
        return s.replace("layers.", "L").replace(".", " ")
    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        d = df[(df.model == key) & (df.ablation == "knockout")]
        M = d.pivot_table(index="module", columns="expert", values="dacc")
        M = M.loc[sorted(M.index, key=lambda s: (s.split(".")[1] if s.startswith("layers") else "0", s))]
        im = ax.imshow(M.values, cmap="Reds_r", vmin=-0.2, vmax=0, aspect="auto"); ax.grid(False)
        ax.set_xticks(range(M.shape[1]), M.columns, fontsize=7)
        ax.set_yticks(range(M.shape[0]), [short(s) for s in M.index], fontsize=7.5)
        for (r, c), v in np.ndenumerate(M.values):
            if not np.isnan(v):
                ax.text(c, r, f"{v * 100:+.0f}", ha="center", va="center", fontsize=5.8, color="white" if v < -0.1 else INK)
        ax.set_title(label_of(key), loc="left", fontsize=9); ax.set_xlabel("expert knocked out", fontsize=8)
    fig.colorbar(im, ax=fig.axes, shrink=0.5, label="Δ accuracy", location="right")
    ax = fig.add_subplot(gs[1, :])
    rr = df[df.ablation == "random routing"].groupby("model")[["dacc", "kl"]].mean().reindex(keys)
    ax.bar(range(len(keys)), rr.dacc, color=[style(k)[0] for k in keys])
    for i, (v, k) in enumerate(zip(rr.dacc, rr.kl)):
        ax.text(i, v - 0.01, f"{v * 100:+.1f} pts\nKL {k:.2f}", ha="center", va="top", fontsize=7)
    ax.set_xticks(range(len(keys)), [label_of(k) for k in keys]); ax.set_ylabel("Δ accuracy")
    ax.set_title("random routing (learned assignment removed, experts kept)", loc="left")
    ax.set_ylim(min(-0.3, rr.dacc.min() - 0.06), 0.02)
    fig.savefig(FIG / fname, bbox_inches="tight"); plt.show()

fig_expert_ablation(run_expert_ablation())
''')

# ---------------------------------------------------------------------------------------------- 12
md(r"""## Unit 12: Ablation: MoM routing (forced SKIP, random routes, module knockout)

MoM's GRU routers pick one attention module and one FFN module (or SKIP) per token at each of the
4 assembly steps of the MoTBlock. We edit the router scores: **forced SKIP** sets the SKIP class to $+10^4$ (the
block becomes the identity, leaving the bottom and anchor layers), **random** draws scores from a Gaussian, and
**knockout** sets module $j$'s score to $-10^4$. Because the MoM checkpoint never learned the task, we also report
KL to its own unablated predictions, which measures how much its output depends on each unit.""")
code(r'''
class RouterEdit:
    def __init__(self, model, fn, which=("attn", "ffn")):
        self.model, self.fn, self.which = model, fn, which
        self.handles = []

    def __enter__(self):
        for n, m in self.model.named_modules():
            if n.endswith("mot_attn.router") or n.endswith("mot_ffn.router"):
                kind = "attn" if "mot_attn" in n else "ffn"
                if kind in self.which:
                    self.handles.append(m.register_forward_hook(lambda mod, inp, out: (self.fn(out[0]),) + tuple(out[1:])))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

def run_mom_ablation(key="mom_ref6_cls_fromfull", force=False):
    path = OUT / "ablation_mom.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    exs = read_examples("test_iid", S["n_ablate"], seed=31)
    m = load_model(key); P0, Y = probs_on(m, exs); acc0 = (P0.argmax(-1) == Y).float().mean().item()
    nmod = m.config.num_attn
    def setcol(j, v):
        def f(s):
            s = s.clone(); s[:, j] = v; return s
        return f
    cases = [("forced SKIP (attn+ffn)", setcol(-1, 1e4), ("attn", "ffn")),
             ("forced SKIP (ffn)", setcol(-1, 1e4), ("ffn",)),
             ("never SKIP", setcol(-1, -1e4), ("attn", "ffn"))]
    cases += [(f"knockout attn {j}", setcol(j, -1e4), ("attn",)) for j in range(nmod)]
    cases += [(f"knockout ffn {j}", setcol(j, -1e4), ("ffn",)) for j in range(nmod)]
    g = torch.Generator().manual_seed(0)
    cases += [("random routes", lambda s: torch.randn(s.shape, generator=g), ("attn", "ffn"))]
    rows = [dict(model=key, ablation="none", acc=acc0, kl=0.0)]
    for name, fn, which in cases:
        with RouterEdit(m, fn, which):
            P, _ = probs_on(m, exs)
        rows.append(dict(model=key, ablation=name, acc=(P.argmax(-1) == Y).float().mean().item(), kl=kl(P0, P)))
    donor = load_model("mom_donor6_cls_full"); Pd, _ = probs_on(donor, exs)
    rows.append(dict(model="mom_donor6_cls_full", ablation="donor (dense, no routing)",
                     acc=(Pd.argmax(-1) == Y).float().mean().item(), kl=kl(P0, Pd)))
    # constant predictor and majority rate for reference
    rows.append(dict(model="-", ablation="majority class", acc=(Y == torch.bincount(Y).argmax()).float().mean().item(), kl=float("nan")))
    df = pd.DataFrame(rows); df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    MOMA = run_mom_ablation()
    display(MOMA.round(4))
    st = load_states("mom_ref6_cls_fromfull")["test"]
    usage = {}
    for b in st["routes"]:
        for step, (mods, act) in b["steps"].items():
            for mod, ids in mods.items():
                usage.setdefault((mod, step), []).append(ids[act])
    print("MoM module usage per step (last id = SKIP):")
    for (mod, step), v in sorted(usage.items()):
        print(f"  {mod:12s} step {step}: {np.bincount(torch.cat(v).numpy(), minlength=5) / sum(len(t) for t in v)}")
''')

md("### Unit 12a: Figure: MoM ablations")
code(r'''
def fig_mom_ablation(df, fname="a12a_mom_ablation.png"):
    d = df[df.ablation != "none"].copy()
    base = df[df.ablation == "none"].acc.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    y = np.arange(len(d))
    cols = [FAMILY_COLOR["MoM"] if m.startswith("mom_ref") else (FAMILY_COLOR["MoM donor"] if m.startswith("mom_donor") else INK2) for m in d.model]
    axes[0].barh(y, d.acc, color=cols); axes[0].axvline(base, color=INK, ls=":", lw=1)
    axes[0].text(base, -0.9, " unablated MoM", fontsize=7)
    axes[0].set_xlim(0.4, 0.62); axes[0].set_title("accuracy", loc="left")
    axes[1].barh(y, d.kl.fillna(0), color=cols); axes[1].set_title("KL(unablated ‖ ablated) of the output", loc="left")
    axes[0].set_yticks(y, d.ablation, fontsize=7.5); axes[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_mom_ablation(run_mom_ablation())
''')

# ---------------------------------------------------------------------------------------------- 13
md(r"""## Unit 13: Training-time ablations (from the pilot runs)

The SUT paper ablates its components by retraining (MIM loss, halting, number of experts); so did the pilot.
This unit collects those runs from `runs/pilot/length/*/result.json` (all seeds) so the inference-time ablations
above can be read next to them.""")
code(r'''
TRAIN_ABL = [  # run name, description
    ("dense6", "Dense (reference)"), ("ut_fixed6", "UT: shared dense block, fixed depth"),
    ("sut_e1_6", "SUT with E=1 (no experts)"), ("sut_fixed6", "SUT without halting"),
    ("sut_common6", "SUT (released MIM sign)"), ("sut_common6_mim_paper", "SUT, MIM paper sign"),
    ("sut_common6_lb", "SUT, load balancing instead of MIM"), ("sut_common6_long", "SUT, 3x training budget"),
    ("mom_donor6_cls", "MoM donor (dense, GPT-2 style)"), ("mom_ref6_cls", "MoM from donor"),
    ("mom_scratch6_cls", "MoM from scratch"), ("donor6_continue_cls", "donor trained further (same budget)"),
]

def collect_training_ablations():
    rows = []
    for run, desc in TRAIN_ABL:
        for d in sorted(RUN_ROOT.glob(f"{run}_s*")):
            if not d.name[len(run):].startswith("_s"):
                continue
            r = json.loads((d / "result.json").read_text())
            t = r.get("test") or {}
            rows.append(dict(run=run, desc=desc, seed=d.name.split("_s")[-1], valid=r["valid"]["acc"],
                             test_iid=t.get("test_iid", {}).get("acc", np.nan), test_len=t.get("test_len", {}).get("acc", np.nan)))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    TA = collect_training_ablations()
    display(TA.groupby(["run", "desc"])[["valid", "test_iid", "test_len"]].agg(["mean", "count"]).round(3))
''')

md("### Unit 13a: Figure: training-time ablations")
code(r'''
def fig_training_ablations(df, fname="a13a_training_ablations.png"):
    order = [r for r, _ in TRAIN_ABL if r in set(df.run)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fam = {"dense6": "Dense", "ut_fixed6": "UT", "mom_donor6_cls": "MoM donor", "donor6_continue_cls": "MoM donor",
           "mom_ref6_cls": "MoM", "mom_scratch6_cls": "MoM", "sut_fixed6": "SUT fixed"}
    for i, r in enumerate(order):
        v = df[df.run == r].valid.values
        col = FAMILY_COLOR.get(fam.get(r, "SUT"), INK2)
        ax.scatter(v, [i] * len(v), color=col, s=22, alpha=0.6, linewidths=0, zorder=3)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=col, lw=2.5, zorder=4)
        ax.annotate(f"{v.mean():.3f}", (v.max(), i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(order)), [dict(TRAIN_ABL)[r] for r in order]); ax.invert_yaxis()
    ax.axvline(0.54, color=INK2, ls=":", lw=1); ax.text(0.541, -0.75, "majority class", fontsize=7, color=INK2)
    ax.set_xlabel("validation accuracy (dots = seeds, bar = mean)")
    ax.set_title("component ablations by retraining (pilot budget: 3,000 steps; MoM rows 10,000)", loc="left")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.show()

fig_training_ablations(collect_training_ablations())
''')

# ---------------------------------------------------------------------------------------------- 14
md(r"""## Unit 14: Synthesis

Collects the key numbers from Units 5–13 into one table (saved to `runs/analysis/<profile>/summary.json`) and
prints the findings they support.""")
code(r'''
def synthesis():
    pr = run_probes(); pt = run_patching(); dep = run_depth_truncation(); thr = run_threshold_sweep()
    ex = run_expert_ablation(); sae = run_sae(); mom = run_mom_ablation(); rt, _ = routing_stats()
    rows = []
    for key in MODELS:
        p = pr[(pr.model == key) & (pr.split == "test")]
        def last(probe):
            d = p[p.probe == probe].sort_values("step"); return d.bacc.iloc[-1] if len(d) else np.nan
        def first_reach(probe, chance=0.5, frac=0.9):
            # first step at which the probe has 90% of its final gain over chance
            d = p[p.probe == probe].sort_values("step")
            if not len(d):
                return np.nan
            gain = d.bacc - chance
            return int(d[gain >= frac * gain.iloc[-1]].step.iloc[0])
        dd = dep[(dep.model == key) & (dep.split == "test")]
        row = dict(model=label_of(key), acc=load_states(key)["test"]["logits"].argmax(-1).numpy().__eq__(
                       build_targets(split_examples()["test"])["label"]).mean(),
                   probe_label=last("label"), probe_region=last("region"), probe_ttA_cls=last("tt(A)@cls"),
                   probe_ttA_tokens=p[p.probe == "tt(A)@A"].bacc.max() if len(p) else np.nan,
                   step_90pct_region=first_reach("region"), PR_last=last("PR"), cos_last=last("cos"),
                   acc_after_2_steps=dd[dd.steps == 2].acc.iloc[0] if len(dd) else np.nan,
                   cls_patch_peak=pt[(pt.model == key) & (pt.group == "<cls>")].recovery.max() if key in set(pt.model) else np.nan,
                   sae_region_r=np.mean([sae[key]["concept_cover"][f"r{i}"] for i in range(1, 5)]) if key in sae else np.nan,
                   routing_entropy=rt[rt.model == key].H_norm.mean() if key in set(rt.model) else np.nan)
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("model")
    save_json(json.loads(tab.round(4).to_json(orient="index")), "summary.json")
    return tab

if __name__ == "__main__":
    SYN = synthesis()
    display(SYN.round(3))
''')

# ---------------------------------------------------------------------------------------------- 15
md(r"""## Unit 15: Figures for the slides

Larger-type versions of the key figures, sized for a 16:9 slide, plus a redrawn schematic of the two
architectures (drawn after SUT Fig. 1/3 and MoM Fig. 1; used only when the original figures are not
available). They read only the cached results of Units 5–13.""")
code(r"""
SLIDE = dict(bg="#f7f5f0", ink="#1d1d1b", mute="#5d5b55", grid="#e2dfd7", accent="#c2410c")
SLIDE_DIR = ROOT / "reports" / "slides" / "figures"; SLIDE_DIR.mkdir(parents=True, exist_ok=True)
SLIDE_KEYS = ["dense6", "switch6", "sut_fixed6", "sut_common6", "sut_common6_long", "mom_ref6_cls_fromfull"]

def slide_style():
    plt.rcParams.update({"figure.facecolor": SLIDE["bg"], "axes.facecolor": SLIDE["bg"], "savefig.facecolor": SLIDE["bg"],
                         "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 12.5, "axes.edgecolor": SLIDE["grid"],
                         "grid.color": SLIDE["grid"], "axes.labelcolor": SLIDE["mute"], "xtick.color": SLIDE["mute"],
                         "ytick.color": SLIDE["mute"], "lines.linewidth": 2.6, "savefig.dpi": 200})

def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center", fontsize=12.5, color=color, weight="bold")
""")

md("### Unit 15a: Slide figure: probing and patching (what is represented, and when it reaches `<cls>`)")
code(r"""
def slide_fig_probing(fname="s_probe_patch.png"):
    slide_style()
    pr = run_probes(); pt = run_patching()
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.2))
    panels = [(axes[0], "label", "label from <cls>  (chance 1/7)", 1 / 7), (axes[1], "region", "region bits $r_1..r_4$ from <cls>  (chance 0.5)", 0.5)]
    for ax, probe, title, chance in panels:
        for k in SLIDE_KEYS + ["dense6@rand"]:
            d = pr[(pr.model == k) & (pr.probe == probe) & (pr.split == "test")].sort_values("step")
            c, ls = style(k)
            ax.plot(d.step, d.bacc, color=c, ls=ls, marker="o", ms=4, alpha=0.75 if k in CONTROLS else 1)
            if probe == "region":
                pass
        ax.axhline(chance, color=SLIDE["mute"], ls=":", lw=1.2)
        ax.set_title(title, loc="left", color=SLIDE["ink"]); ax.set_xlabel("step  (−1 = embeddings)")
        ax.set_xticks(range(-1, 6))
    axes[0].set_ylabel("balanced accuracy (test)")
    ax = axes[2]
    for k in SLIDE_KEYS:
        d = pt[(pt.model == k) & (pt.group == "<cls>")].sort_values("step")
        if not len(d) or d.n_pairs.iloc[0] < 20:
            continue
        c, ls = style(k)
        ax.plot(d.step, d.recovery, color=c, ls=ls, marker="o", ms=4)
    ax.set_title("patching: recovered by <cls> alone", loc="left", color=SLIDE["ink"]); ax.set_xlabel("patched step")
    ax.set_xticks(range(-1, 6)); ax.set_ylabel("recovered fraction $R$")
    handles = [plt.Line2D([], [], color=style(k)[0], ls=style(k)[1], lw=2.6, marker="o", ms=4) for k in SLIDE_KEYS + ["dense6@rand"]]
    fig.legend(handles, [label_of(k) for k in SLIDE_KEYS + ["dense6@rand"]], loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=12.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(SLIDE_DIR / fname); plt.show()

slide_fig_probing()
""")

md("### Unit 15b: Slide figure: ablations (depth truncation, halting threshold, routing knockouts)")
code(r"""
def slide_fig_ablations(fname="s_ablations.png"):
    slide_style()
    dep = run_depth_truncation(); thr = run_threshold_sweep(); ex = run_expert_ablation(); mom = run_mom_ablation()
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1, 1.1]))
    ax = axes[0]
    for k in ["dense6", "ut_fixed6", "switch6", "sut_fixed6", "sut_common6", "sut_common6_long"]:
        d = dep[(dep.model == k) & (dep.split == "test")]
        c, ls = style(k)
        ax.plot(d.steps, d.acc, color=c, ls=ls, marker="o", ms=4, label=label_of(k))
    ax.axhline(0.54, color=SLIDE["mute"], ls=":", lw=1.2); ax.text(1, 0.545, "majority class", ha="left", va="bottom", fontsize=11, color=SLIDE["mute"])
    ax.set_title("depth truncation: run only k steps", loc="left", color=SLIDE["ink"]); ax.set_xlabel("steps executed k")
    ax.set_ylabel("accuracy (test)"); ax.legend(fontsize=10.5, loc="lower right", ncol=1)
    ax = axes[1]
    for k in thr.model.unique():
        d = thr[(thr.model == k) & (thr.split == "test")]
        c, ls = style(k)
        ax.plot(d.mean_depth, d.acc, color=c, ls=ls, marker="o", ms=5, label=label_of(k))
        r = d[d.tau == 0.999].iloc[0]
        ax.scatter([r.mean_depth], [r.acc], s=120, facecolors="none", edgecolors=c, lw=2, zorder=5)
        ax.annotate(label_of(k), (d.mean_depth.min(), d.acc.iloc[0]), xytext=(0, 12), textcoords="offset points",
                    fontsize=12, color=c, weight="bold")
    ax.set_title("halting threshold sweep (circle = trained τ)", loc="left", color=SLIDE["ink"])
    ax.set_xlabel("mean expected halting step (τ from 0.5 to 1)"); ax.set_ylabel("accuracy (test)")
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.01)
    ax = axes[2]
    rows = []
    for k in ["switch6", "moa6", "sut_fixed6", "sut_common6", "sut_common6_long"]:
        d = ex[ex.model == k]
        rows.append((label_of(k), d[d.ablation == "knockout"].dacc.min(), d[d.ablation == "random routing"].dacc.mean(), style(k)[0]))
    m0 = mom[mom.ablation == "none"].acc.iloc[0]
    rows.append(("MoM", (mom[mom.ablation.str.startswith("knockout")].acc - m0).min(),
                 (mom[mom.ablation == "random routes"].acc.iloc[0] - m0), style("mom_ref6_cls_fromfull")[0]))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r[1] * 100 for r in rows], height=0.38, color=[r[3] for r in rows], alpha=0.55, label="worst single-expert knockout")
    ax.barh(y + 0.2, [r[2] * 100 for r in rows], height=0.38, color=[r[3] for r in rows], label="random routing")
    for i, r in enumerate(rows):
        ax.text(min(r[2] * 100, 0) - 0.5, i + 0.2, f"{r[2] * 100:+.1f}", va="center", ha="right", fontsize=11.5, color=SLIDE["ink"])
    ax.set_yticks(y, [r[0] for r in rows]); ax.invert_yaxis(); ax.axvline(0, color=SLIDE["ink"], lw=1)
    ax.set_xlabel("Δ accuracy (points)"); ax.set_title("routing ablations", loc="left", color=SLIDE["ink"])
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=SLIDE["mute"], alpha=0.45), plt.Rectangle((0, 0), 1, 1, color=SLIDE["mute"])],
              ["worst single-expert knockout", "random routing"], fontsize=11, loc="lower left"); ax.set_xlim(min(-35, min(r[2] for r in rows) * 100 - 8), 3)
    fig.tight_layout(); fig.savefig(SLIDE_DIR / fname); plt.show()

slide_fig_ablations()
""")

md("### Unit 15c: Slide figure: redrawn schematics of SUT and MoM (fallback for the paper figures)")
code(r"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def _box(ax, x, y, w, h, text, fc, ec=None, fs=12, tc=None, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025", fc=fc, ec=ec or fc, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or SLIDE["ink"])

def _arrow(ax, a, b, color=None, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color or SLIDE["mute"], lw=lw, ls=ls))

def slide_fig_schematics(fname="s_schematics.png"):
    slide_style()
    fig, axes = plt.subplots(1, 2, figsize=(16.6, 5.4), gridspec_kw=dict(width_ratios=[1, 1.08]))
    blue, orange, green, grey, purple = "#d6e6f8", "#fbe0d2", "#d5f0e4", "#e9e6df", "#e4e0f7"
    # --- SUT: one shared block applied repeatedly, experts chosen per token, stick-breaking halting
    ax = axes[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("SUT: one shared sparse block, applied up to L times", loc="left", color=SLIDE["ink"])
    for i, y in enumerate([0.08, 0.36, 0.64]):
        _box(ax, 0.06, y, 0.5, 0.22, "", "#ffffff", ec=SLIDE["grid"])
        _box(ax, 0.09, y + 0.125, 0.44, 0.065, "MoMHA: top-k of E query/output experts", blue, fs=10.5)
        _box(ax, 0.09, y + 0.03, 0.44, 0.065, "MoE FFN: top-k of E experts", orange, fs=10.5)
        ax.text(0.65, y + 0.14, f"step {i + 1}" if i < 2 else "step l", fontsize=11, color=SLIDE["mute"], ha="center")
        if i < 2:
            _arrow(ax, (0.31, y + 0.225), (0.31, y + 0.355))
        _box(ax, 0.74, y + 0.06, 0.2, 0.1, r"halt? $\alpha_t^{(l)}$", green, fs=11.5)
        _arrow(ax, (0.56, y + 0.11), (0.74, y + 0.11), color=SLIDE["mute"])
    ax.text(0.84, 0.95, "same weights at every step", ha="center", fontsize=11, color=SLIDE["accent"])
    _arrow(ax, (0.31, 0.0), (0.31, 0.08))
    ax.text(0.33, 0.015, "tokens  <cls>  A  #SEP#  B", fontsize=10.5, color=SLIDE["mute"])
    ax.text(0.84, 0.03, r"$p_t^{(l)}=\alpha_t^{(l)}\prod_{l'<l}(1-\alpha_t^{(l')})$", ha="center", fontsize=12)
    # --- MoM: bottom layer, routed assembly over H steps, anchor layer
    ax = axes[1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("MoM [1-4-1]: modules assembled per token by routers", loc="left", color=SLIDE["ink"])
    _box(ax, 0.05, 0.05, 0.36, 0.11, "bottom layer (dense, from donor)", grey, fs=11)
    _box(ax, 0.05, 0.80, 0.36, 0.11, "anchor layer (dense, from donor)", grey, fs=11)
    _box(ax, 0.05, 0.22, 0.36, 0.52, "", "#ffffff", ec=SLIDE["grid"])
    ax.text(0.23, 0.70, "MoTBlock: H = 4 assembly steps", ha="center", fontsize=11.5, color=SLIDE["ink"])
    for j, y in enumerate([0.27, 0.37, 0.47, 0.57]):
        _box(ax, 0.08, y, 0.13, 0.06, f"Attn$_{{a_{j + 1}}}$", blue, fs=11)
        _box(ax, 0.25, y, 0.13, 0.06, f"FFN$_{{f_{j + 1}}}$", orange, fs=11)
        ax.text(0.065, y + 0.03, f"{j + 1}", fontsize=10, color=SLIDE["mute"], ha="right", va="center")
    _arrow(ax, (0.23, 0.16), (0.23, 0.22)); _arrow(ax, (0.23, 0.74), (0.23, 0.80))
    _box(ax, 0.52, 0.52, 0.42, 0.2, "GRU router (state carried over\nthe 4 steps) picks 1 of 4 attention\nand 1 of 4 FFN modules, or SKIP", purple, fs=11)
    _arrow(ax, (0.52, 0.6), (0.41, 0.5), color="#6250d6")
    ax.text(0.53, 0.40, "modules = the 4 middle layers'\nattention / FFN of the dense donor\n(conversion: init_from_vanilla)", fontsize=11, color=SLIDE["mute"], va="top")
    ax.text(0.53, 0.18, "redrawn after Shen et al. 2023 (SUT, Fig. 1, 3)\nand Gong et al. 2024 (MoM, Fig. 1)", fontsize=10, color=SLIDE["mute"], va="top", style="italic")
    fig.tight_layout(); fig.savefig(SLIDE_DIR / fname); plt.show()

slide_fig_schematics()
""")


def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (t, s) in enumerate(CELLS):
        cell = {"cell_type": t, "id": f"a{i:02d}", "metadata": {}, "source": s.strip("\n").splitlines(True)}
        if t == "code":
            cell.update(outputs=[], execution_count=None)
        nb["cells"].append(cell)
    Path(path).write_text(json.dumps(nb, indent=1), encoding="utf-8")


if __name__ == "__main__":
    build(Path(__file__).with_name("MoE_Analysis.ipynb"))
    print("written", len(CELLS), "cells")
