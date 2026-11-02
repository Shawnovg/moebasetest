"""Routing analysis on the saved usage counts (result.json -> routing).

* ``task_unit_matrix(row)``  task x unit count matrix for one module of one run
* ``mutual_information(M)``  I(task; unit) / H(task) in [0, 1]: how much the unit choice reveals the task
  (0 = every task uses units in the same proportions, 1 = the unit identifies the task)
* ``collapse_table(routing_df)`` per structure x balance mode: mean entropy, max load, dead units
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def task_unit_matrix(row):
    tasks = sorted(row["per_task"])
    return tasks, np.array([row["per_task"][t] for t in tasks], dtype=float)


def mutual_information(M: np.ndarray) -> float:
    P = M / M.sum()
    pt, pu = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / (pt @ pu)[nz])).sum())
    ht = float(-(pt[pt > 0] * np.log(pt[pt > 0])).sum())
    return mi / ht if ht > 0 else 0.0


def collapse_table(rdf: pd.DataFrame) -> pd.DataFrame:
    return (rdf.groupby(["dataset", "structure", "variant", "balance"])
            .agg(entropy=("entropy", "mean"), min_entropy=("entropy", "min"), max_load=("max_load", "mean"),
                 dead=("dead", "mean"), n_modules=("module", "nunique"))
            .reset_index())


def specialization_table(rdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in rdf.iterrows():
        if len(r["per_task"]) < 2:
            continue
        _, M = task_unit_matrix(r)
        rows.append({k: r[k] for k in ("dataset", "structure", "variant", "balance", "seed", "module")} |
                    {"task_mi": mutual_information(M)})
    return pd.DataFrame(rows)
