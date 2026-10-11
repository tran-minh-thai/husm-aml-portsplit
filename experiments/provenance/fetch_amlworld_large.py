"""Download the AMLworld HI-Large and LI-Large files from Kaggle and check them against the manifest.

The files go to datasets/amlworld/ next to the Small and Medium files. Each one is downloaded to a
".part" file, its size and sha256 are compared with docs/reproduce/datasets_large.sha256, and only a
matching file gets its final name. A file already present with the right sha256 is not downloaded
again. The Kaggle API key is read from KAGGLE_USERNAME and KAGGLE_KEY, or from kaggle.json in the
.kaggle directory of the home directory; it is never printed. Standard library only (runs on Windows).

Run: python experiments/provenance/fetch_amlworld_large.py [--check-only]
Exit status 0 only if every file in the manifest is present and matches.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import shutil
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "reproduce" / "datasets_large.sha256"
DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"


def read_manifest(path: pathlib.Path) -> list[tuple[str, int, str]]:
    """(sha256, size, path relative to the repository root) per line."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, size, rel = line.split(maxsplit=2)
            rows.append((digest, int(size), rel))
    return rows


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def matches(path: pathlib.Path, size: int, digest: str) -> str | None:
    """None when the file has this size and digest, otherwise why not."""
    if not path.is_file():
        return f"{path.name}: missing"
    if path.stat().st_size != size:
        return f"{path.name}: size {path.stat().st_size}, manifest {size}"
    got = sha256(path)
    if got != digest:
        return f"{path.name}: sha256 {got}, manifest {digest}"
    return None


def credentials() -> str:
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        conf = pathlib.Path.home() / ".kaggle" / "kaggle.json"
        if not conf.is_file():
            sys.exit(f"no Kaggle key: set KAGGLE_USERNAME and KAGGLE_KEY, or put kaggle.json at {conf}")
        c = json.loads(conf.read_text(encoding="utf-8"))
        user, key = c["username"], c["key"]
    return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()


def download(name: str, target: pathlib.Path, auth: str) -> None:
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}/{name}"
    req = urllib.request.Request(url, headers={"Authorization": auth})
    with urllib.request.urlopen(req, timeout=120) as r:
        ctype = r.headers.get("Content-Type", "")
        if "zip" in ctype:
            sys.exit(f"{name}: the server sent a zip archive ({ctype}); this script expects the plain file")
        with open(target, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)


def fetch(rows, root: pathlib.Path, check_only: bool, get=download) -> list[str]:
    """Problems left after fetching; empty when every file matches."""
    problems, auth = [], None
    for digest, size, rel in rows:
        final = root / rel
        if matches(final, size, digest) is None:
            print(f"[fetch] {rel}: present, matches")
            continue
        if check_only:
            problems.append(matches(final, size, digest))
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        part = final.with_name(final.name + ".part")
        auth = auth or credentials()
        print(f"[fetch] {rel}: downloading {size} bytes", flush=True)
        get(final.name, part, auth)
        why = matches(part, size, digest)
        if why:
            problems.append(f"{rel}: downloaded file does not match ({why}); kept as {part.name}")
            continue
        os.replace(part, final)
        print(f"[fetch] {rel}: downloaded, matches")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="only check files already present")
    args = ap.parse_args()
    rows = read_manifest(MANIFEST)
    problems = fetch(rows, ROOT, args.check_only)
    for p in problems:
        print(f"[fetch] MISMATCH {p}")
    print(f"[fetch] {len(rows) - len(problems)} of {len(rows)} files match the manifest")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
