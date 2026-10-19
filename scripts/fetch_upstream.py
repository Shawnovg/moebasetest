"""Clone the audited upstream repositories at their pinned commits into third_party/.

    python scripts/fetch_upstream.py            # git clone + checkout (falls back to zip download)

Nothing from SUT or MoM is vendored into this repository (neither ships a LICENSE file); the code in
src/moebase is an independent port that cites upstream line-by-line, and the parity tests load the
pinned upstream files from third_party/ to compare numerically.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moebase import paths  # noqa: E402


def have_git() -> bool:
    return shutil.which("git") is not None


def fetch(name: str, url: str, commit: str):
    dest = paths.upstream_dir(name)
    if dest.exists() and not any(dest.iterdir()):
        dest.rmdir()  # empty leftover from an interrupted clone
    pin = dest / "PINNED_COMMIT"
    if pin.exists():
        found = pin.read_text().strip()
        print(f"{name}: pinned copy present ({found[:8]})" + ("" if found == commit else f"  WARNING expected {commit[:8]}"))
        return
    if dest.exists():
        if (dest / ".git").exists() and have_git():
            head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            if head == commit:
                print(f"{name}: present at {commit[:8]}")
                return
            subprocess.run(["git", "-C", str(dest), "fetch", "--quiet"], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
            print(f"{name}: checked out {commit[:8]}")
            return
        print(f"{name}: {dest} exists (not a git checkout); leaving it untouched")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if have_git():
        subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
        print(f"{name}: cloned at {commit[:8]}")
        return
    zurl = url.replace("https://github.com/", "https://codeload.github.com/") + f"/zip/{commit}"
    print(f"{name}: git not found, downloading {zurl}")
    data = urllib.request.urlopen(zurl).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(dest.parent)
    (dest.parent / top).rename(dest)
    (dest / "PINNED_COMMIT").write_text(commit)
    print(f"{name}: extracted at {commit[:8]}")


def main():
    for name, info in paths.UPSTREAM.items():
        fetch(name, info["url"], info["commit"])
    print("License notes:")
    for name, info in paths.UPSTREAM.items():
        print(f"  {name}: {info['license']}")


if __name__ == "__main__":
    main()
