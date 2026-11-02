"""Execute a notebook top to bottom without a Jupyter kernel and save it with outputs.

    python scripts/run_notebook.py notebooks/MoE_Experiment.ipynb [--out executed.ipynb] [--cells 0-5]

Captures printed text, display() of DataFrames (text + HTML) and matplotlib figures (PNG), so the
saved notebook opens in Jupyter/VS Code with results visible. Stops at the first failing cell and
records the traceback in that cell.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _patch_ipython(display):
    try:
        import IPython.display as ipd
        ipd.display = display
    except ImportError:
        mod = type(sys)("IPython.display")
        mod.display = display
        pkg = type(sys)("IPython")
        pkg.display = mod
        pkg.get_ipython = lambda: None  # matplotlib probes this
        sys.modules["IPython"] = pkg
        sys.modules["IPython.display"] = mod


def run(nb_path: Path, out_path: Path, cell_range=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    os.chdir(nb_path.parent)
    glb = {"__name__": "__main__"}
    current = []

    def display(obj):
        data = {"text/plain": obj.to_string() if hasattr(obj, "to_string") else repr(obj)}
        if hasattr(obj, "to_html"):
            data["text/html"] = obj.to_html()
        current.append({"output_type": "display_data", "data": data, "metadata": {}})

    def show(*a, **k):
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", bbox_inches="tight")
            current.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": "<Figure>"}})
        plt.close("all")

    glb["display"] = display  # the notebook keeps this when IPython is not installed
    plt.show = show
    code_idx = -1
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        code_idx += 1
        if cell_range and not (cell_range[0] <= code_idx <= cell_range[1]):
            continue
        current.clear()
        src = "".join(cell["source"])
        stdout = io.StringIO()
        t0 = time.time()
        ok = True
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            try:
                exec(compile(src, f"<cell {code_idx}>", "exec"), glb)
                show()
            except Exception:  # noqa: BLE001
                ok = False
                err = traceback.format_exc()
        count += 1
        outs = []
        if stdout.getvalue():
            outs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue()})
        outs += list(current)
        if not ok:
            outs.append({"output_type": "error", "ename": "Error", "evalue": err.strip().splitlines()[-1],
                         "traceback": err.splitlines()})
        cell["outputs"] = outs
        cell["execution_count"] = count
        print(f"cell {code_idx:2d} {'ok ' if ok else 'ERR'} {time.time() - t0:7.1f}s", flush=True)
        out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        if not ok:
            print(err)
            return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cells", default=None)
    a = ap.parse_args()
    rng = tuple(int(x) for x in a.cells.split("-")) if a.cells else None
    nbp = a.notebook.resolve()
    sys.exit(run(nbp, (a.out or nbp).resolve(), rng))
