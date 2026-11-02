"""Regenerate every table (reports/tables/*.csv) and figure (reports/figures/*.png) used in the report.

    python -m analysis.report_tables
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from analysis import figures as F  # noqa: E402
from analysis.collect import agg, load_history, load_results, load_routing, paired_delta, separation  # noqa: E402
from analysis.routing import collapse_table, specialization_table  # noqa: E402
from analysis.schematics import SPECS, draw  # noqa: E402
from backbone.paths import PROJECT_ROOT, RUNS_ROOT  # noqa: E402

TAB = PROJECT_ROOT / "reports" / "tables"


def chosen():
    p = PROJECT_ROOT / "configs" / "chosen_datasets.json"
    return json.loads(p.read_text()) if p.exists() else {"main": [], "variants": []}


def build_all(runs_root=RUNS_ROOT):
    TAB.mkdir(parents=True, exist_ok=True)
    out = []
    df = load_results(runs_root)
    df.to_csv(TAB / "all_runs.csv", index=False); out.append(TAB / "all_runs.csv")
    hist, rdf = load_history(runs_root), load_routing(runs_root)
    sep = separation(df)
    sep.to_csv(TAB / "screening_separation.csv", index=False)
    scr = df[(df.variant == "default") & (df.balance == "none")]
    scr.pivot_table(index="structure", columns="dataset", values="test_acc").to_csv(TAB / "screening_acc.csv")
    out += [F.screen_heatmap(df), F.screen_separation(sep)]
    agg(df, ["dataset", "structure", "variant", "balance"]).to_csv(TAB / "acc_by_config.csv", index=False)
    deltas = pd.concat([paired_delta(df, "lb", "none"), paired_delta(df, "warmup", "none"),
                        paired_delta(df, "lb", "warmup")])
    deltas = deltas[deltas["count"] > 0]
    deltas.to_csv(TAB / "balance_deltas.csv", index=False)
    ct = collapse_table(rdf) if len(rdf) else pd.DataFrame()
    ct.to_csv(TAB / "routing_collapse.csv", index=False)
    sp = specialization_table(rdf) if len(rdf) else pd.DataFrame()
    sp.to_csv(TAB / "task_specialization.csv", index=False)
    ch = chosen()
    for d in sorted(set(ch["main"]) | set(ch["variants"])):
        out += [F.main_bars(df, d), F.balance_delta(deltas[deltas.variant == "default"], d), F.collapse_overview(ct, d)]
        for b in ("none", "lb", "warmup"):
            out.append(F.curves(hist, d, b))
        for s in ["moe", "moh", "moa", "momha", "mom", "mita"]:
            if len(df[(df.dataset == d) & (df.structure == s) & (df.variant != "default")]):
                out.append(F.variants_bars(df, d, s))
                out.append(F.balance_delta(deltas[(deltas.structure == s)], d, name=f"balance_delta_{d}_{s}"))
            for v in sorted(set(rdf[(rdf.dataset == d) & (rdf.structure == s)].variant)) if len(rdf) else []:
                out.append(F.routing_load(rdf, d, s, v))
    if len(rdf):
        mix = rdf[(rdf.dataset == "mix") & (rdf.variant == "default") & (rdf.seed == 0) & (rdf.balance == "none")]
        for s in sorted(set(mix.structure)):
            row = mix[mix.structure == s].iloc[-1]
            out.append(F.task_unit_heatmap(row, f"task_unit_mix_{s}"))
    out += [draw(s) for s in SPECS]
    return [p for p in out if p]


if __name__ == "__main__":
    for p in build_all():
        print(p)
