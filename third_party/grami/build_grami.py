"""Build the GraMi variants used as a same-output speed baseline, on any operating system.

    grami-path                      GraMi restricted to directed paths (grami-path.patch)
    grami-path-nocache              the same with GraMi's own constant Settings.CACHING set to false
                                    (grami-nocache.patch): with caching on, GraMi imports values found not to
                                    extend from cached patterns one edge smaller, and on AMLworld under
                                    labeling A this makes it miss frequent directed paths
    grami-path-sharedtimer          grami-path with one shared timer thread for the five-second stop of every
    grami-path-nocache-sharedtimer  search instead of a new java.util.Timer thread per search
                                    (grami-sharedtimer.patch); the stop and the output are unchanged
    grami-path-nocache-sharedtimer-dump
                                    the previous one that, with -Dgrami.dump=<edges>, also writes the values
                                    left in the domain of every variable of each frequent pattern with that
                                    many edges to grami-dump.txt (grami-dump.patch; diagnosis only, not timed)
    grami-path-nocache-sharedtimer-acfix
                                    grami-path-nocache-sharedtimer whose arc consistency (DFSSearch.AC_3_New) also
                                    revises the arcs leaving a variable whose domain shrank, not only the arcs
                                    entering it (grami-acfix.patch); GraMi decides patterns with distinct vertex
                                    labels and no cycle by arc consistency alone

Applies the patches of this directory by exact replacement (no patch program needed; line endings of the
checkout do not matter), refuses a GraMi checkout other than the pinned commit, and compiles each variant with
javac into <out>/<variant>/bin, together with the two empty stand-ins of stubs/.

Run: python third_party/grami/build_grami.py --grami <clone of GraMi> --out <directory>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMIT = "d2875ae80afa2f141323a3f6b89a387787fcda38"
SRC = "GRAMI_DIRECTED_SUBGRAPHS/src"
VARIANTS = {"grami-path": ["grami-path.patch"], "grami-path-nocache": ["grami-path.patch", "grami-nocache.patch"],
            "grami-path-sharedtimer": ["grami-path.patch", "grami-sharedtimer.patch"],
            "grami-path-nocache-sharedtimer": ["grami-path.patch", "grami-nocache.patch", "grami-sharedtimer.patch"],
            "grami-path-nocache-sharedtimer-dump": ["grami-path.patch", "grami-nocache.patch", "grami-sharedtimer.patch",
                                                   "grami-dump.patch"],
            "grami-path-nocache-sharedtimer-acfix": ["grami-path.patch", "grami-nocache.patch", "grami-sharedtimer.patch",
                                                    "grami-acfix.patch"]}


def hunks(patch: Path):
    """(path below GRAMI_DIRECTED_SUBGRAPHS/src, old text or None for a new file, new text) per file."""
    lines = patch.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    files, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("+++ "):
            i += 1
            continue
        target = lines[i][4:].split("\t")[0]
        rel = target.split(SRC + "/", 1)[1]
        new_file = lines[i - 1].startswith("--- ") and "1970-01-01" in lines[i - 1]
        i += 1
        blocks = []
        while i < len(lines) and not lines[i].startswith("diff ") and not lines[i].startswith("--- "):
            if lines[i].startswith("@@"):
                old, new = [], []
                i += 1
                while i < len(lines) and lines[i][:1] in (" ", "-", "+") and not lines[i].startswith(("--- ", "+++ ")):
                    tag, text = lines[i][:1], lines[i][1:]
                    if tag in (" ", "-"):
                        old.append(text)
                    if tag in (" ", "+"):
                        new.append(text)
                    i += 1
                blocks.append(("\n".join(old), "\n".join(new)))
            else:
                i += 1
        files.append((rel, None if new_file else blocks, blocks))
    return files


def apply(src: Path, patch: Path) -> None:
    for rel, old_blocks, blocks in hunks(patch):
        path = src / rel
        if old_blocks is None:
            path.write_text(blocks[0][1] + "\n", encoding="ISO-8859-1")
            continue
        text = path.read_bytes().decode("ISO-8859-1").replace("\r\n", "\n")
        for old, new in old_blocks:
            if text.count(old) != 1:
                sys.exit(f"{patch.name}: the context of a change to {rel} is found {text.count(old)} times, expected once")
            text = text.replace(old, new)
        path.write_bytes(text.encode("ISO-8859-1"))


def javac() -> str:
    """javac on PATH, else from JAVA_HOME (Maven can run with JAVA_HOME set and no javac on PATH)."""
    found = shutil.which("javac")
    if found:
        return found
    home = os.environ.get("JAVA_HOME")
    for name in ("javac.exe", "javac"):
        if home and (Path(home) / "bin" / name).is_file():
            return str(Path(home) / "bin" / name)
    sys.exit("javac not found: not on PATH and not under JAVA_HOME/bin")


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.java")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grami", required=True, help="a clone of https://github.com/ehab-abdelhamid/GraMi")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    grami, out = Path(args.grami), Path(args.out)
    head = subprocess.run(["git", "-C", str(grami), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        sys.exit(f"refusing to build: {grami} is at {head or '(not a git checkout)'}, expected {COMMIT}")
    for variant, patches in VARIANTS.items():
        dest = out / variant
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(grami / SRC, dest / "src")
        for p in patches:
            apply(dest / "src", HERE / p)
        sources = sorted(str(p) for p in (dest / "src").rglob("*.java")) + sorted(str(p) for p in (HERE / "stubs").rglob("*.java"))
        argfile = dest / "sources.txt"
        argfile.write_text("\n".join('"' + s.replace("\\", "/") + '"' for s in sources) + "\n", encoding="utf-8")
        (dest / "bin").mkdir()
        subprocess.run([javac(), "-nowarn", "-encoding", "ISO-8859-1", "-d", str(dest / "bin"), f"@{argfile}"], check=True)
        if not (dest / "bin" / "Dijkstra" / "main.class").is_file():
            sys.exit(f"{variant}: javac did not write Dijkstra/main.class")
        print(f"{variant}: {len(sources)} sources, source tree sha256 {tree_digest(dest / 'src')}")


if __name__ == "__main__":
    main()
