"""Render LaTeX equations to transparent PNGs (pdflatex + standalone + pdftoppm).

    python scripts/render_latex.py equations.json out_dir [--dpi 300] [--color 1a1a1a]

equations.json: {"name": "LaTeX math (display style, no $...$)", ...}
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TEMPLATE = r"""\documentclass[border=2pt]{standalone}
\usepackage{amsmath,amssymb,bm}
\usepackage{xcolor}
\definecolor{ink}{HTML}{%s}
\definecolor{acc}{HTML}{%s}
\begin{document}
\color{ink}
$\displaystyle %s$
\end{document}
"""


def render(name, tex, out_dir, dpi=300, color="1a1a1a", accent="c2410c"):
    with tempfile.TemporaryDirectory() as d:
        Path(d, "eq.tex").write_text(TEMPLATE % (color, accent, tex))
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "eq.tex"], cwd=d,
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{name}: " + "\n".join(l for l in r.stdout.splitlines() if l.startswith("!"))[:800])
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", "eq.pdf", "eq"], check=True, cwd=d)
        png = Path(out_dir, name + ".png")
        to_transparent(Path(d, "eq.png"), png)
    return png


def to_transparent(src, dst):
    """Ink rendered on white -> RGBA with alpha recovered (un-premultiplied over white)."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(src).convert("RGB")).astype(np.float32) / 255
    alpha = 1 - a.min(-1)
    rgb = np.where(alpha[..., None] > 1e-3, (a - (1 - alpha[..., None])) / np.maximum(alpha[..., None], 1e-3), 0)
    out = np.dstack([np.clip(rgb, 0, 1), alpha])
    Image.fromarray((out * 255).round().astype(np.uint8), "RGBA").save(dst)


if __name__ == "__main__":
    eqs = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    dpi = int(sys.argv[sys.argv.index("--dpi") + 1]) if "--dpi" in sys.argv else 300
    for n, t in eqs.items():
        print(render(n, t, out, dpi))
