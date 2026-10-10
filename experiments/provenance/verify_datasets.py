"""Check the benchmark files under datasets/ against the committed checksum manifest.

    python experiments/provenance/verify_datasets.py           # verify every listed file
    python experiments/provenance/verify_datasets.py --write   # (re)write the manifest from datasets/

The manifest, docs/reproduce/datasets.sha256, lists one file per line: sha256, size in bytes,
and the path relative to the repository root with '/' separators. Verification prints one line
per file (OK, MISSING, SIZE or HASH) and exits non-zero unless every listed file matches; files
present under datasets/ but not listed are reported too, since a run may read them.

Standard library only; runs on Windows, macOS and Linux.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"
MANIFEST = ROOT / "docs" / "reproduce" / "datasets.sha256"
IGNORED = {".DS_Store", "Thumbs.db", "desktop.ini", "Icon\r"}


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def present_files(data: Path = DATA) -> list[Path]:
    if not data.is_dir():
        return []
    return sorted((p for p in data.rglob("*") if p.is_file() and p.name not in IGNORED),
                  key=lambda p: p.relative_to(data).as_posix())


def read_manifest(path: Path = MANIFEST) -> list[tuple[str, int, str]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3 or len(parts[0]) != 64 or not parts[1].isdigit():
            raise SystemExit(f"{path.name}:{n}: malformed line: {line!r}")
        rows.append((parts[0], int(parts[1]), parts[2]))
    return rows


def verify(manifest: Path = MANIFEST, root: Path = ROOT) -> int:
    if not manifest.is_file():
        print(f"[verify] no manifest at {manifest}")
        return 2
    rows = read_manifest(manifest)
    bad = 0
    for digest, size, rel in rows:
        path = root / rel
        if not path.is_file():
            status = "MISSING"
        elif path.stat().st_size != size:
            status = f"SIZE {path.stat().st_size} (expected {size})"
        elif sha256(path) != digest:
            status = "HASH"
        else:
            status = "OK"
        bad += status != "OK"
        print(f"[verify] {status:8s} {rel}", flush=True)
    listed = {rel for _, _, rel in rows}
    extra = [p.relative_to(root).as_posix() for p in present_files(root / "datasets")
             if p.relative_to(root).as_posix() not in listed]
    for rel in extra:
        print(f"[verify] UNLISTED {rel}")
    if bad:
        print(f"[verify] {bad} of {len(rows)} files do not match the manifest")
        return 1
    print(f"[verify] all {len(rows)} files match the manifest"
          + (f" ({len(extra)} unlisted file(s) present)" if extra else ""))
    return 0


def write_manifest(manifest: Path = MANIFEST) -> int:
    files = present_files()
    if not files:
        print(f"[write] nothing under {DATA}")
        return 2
    lines = ["# sha256  bytes  path  -- written by experiments/provenance/verify_datasets.py --write"]
    for p in files:
        lines.append(f"{sha256(p)}  {p.stat().st_size}  {p.relative_to(ROOT).as_posix()}")
        print(lines[-1], flush=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[write] {len(files)} files -> {manifest.relative_to(ROOT).as_posix()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the manifest from the files present")
    args = ap.parse_args()
    return write_manifest() if args.write else verify()


if __name__ == "__main__":
    sys.exit(main())
