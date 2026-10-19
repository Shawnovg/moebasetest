"""Build audited logical-inference splits and an immutable manifest.

    python scripts/fetch_upstream.py            # once: clones Ordered-Memory at the pinned commit
    python scripts/prepare_logic_data.py        # writes data/logic-v1/{regime}/{part}.jsonl + manifest.json

Refuses to overwrite an existing version directory unless --force is given (manifests are immutable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moebase import paths  # noqa: E402
from moebase.data import logic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=paths.logic_raw_dir())
    ap.add_argument("--out", type=Path, default=paths.DATA_DIR / paths.DATA_VERSION)
    ap.add_argument("--skip-label-check", action="store_true", help="skip exhaustive truth-table check")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not (args.raw / "train0").exists():
        sys.exit(f"Raw data not found in {args.raw}. Run scripts/fetch_upstream.py first "
                 "or set MOE_BASE_LOGIC_RAW.")
    if (args.out / "manifest.json").exists() and not args.force:
        sys.exit(f"{args.out} already exists; bump DATA_VERSION or pass --force.")

    raw = logic.load_raw(args.raw)
    splits = logic.build_splits(raw)
    report = logic.audit(raw, splits, check_labels=not args.skip_label_check)
    manifest = logic.write_splits(splits, args.out, args.raw, report, paths.DATA_VERSION)

    print(json.dumps({k: v for k, v in report.items() if k != "leakage"}, indent=1))
    print("leakage:", json.dumps(report["leakage"], indent=1))
    for regime, parts in manifest["splits"].items():
        print(regime, {p: d["n"] for p, d in parts.items()})
    if not report["systematicity_sizes_match_paper"]:
        print("WARNING: systematicity split sizes do not match Ordered Memory Table 2")
    if report.get("label_mismatches"):
        print("WARNING: label mismatches found; see manifest")


if __name__ == "__main__":
    main()
