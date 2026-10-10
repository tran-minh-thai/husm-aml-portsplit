"""Compare the records of the same command made on different machines, exactly.

    python experiments/provenance/compare_records.py results-invariant/<question> [...]

Records under a question directory are grouped by their recorded command (the interpreter path
dropped), and within a group every record is compared with the oldest one. For each result file
of the oldest record (run_meta.json and stderr.txt excepted) the file must exist in the other
record and be equal:

  * CSV files cell by cell after reading both with the csv module (so CRLF and LF compare
    equal); cells are compared as strings, with no tolerance, because a difference of 1e-12
    means a random stream or an evaluation order changed, which a tolerance would hide;
  * JSON and JSON-lines files value by value after parsing (numbers compared as exact doubles),
    since the engine's Java runtime may print one double differently across versions;
  * other text files line by line after normalizing line endings, with every path written by
    the recording wrapper (it starts with "<repository root>") and every run id masked;
  * other files byte by byte.

Every comparison prints its denominator (cells, lines or bytes compared). The exit code is 0
only if at least one pair was compared and nothing differed.

Standard library only; runs on Windows, macOS and Linux.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

SKIP = {"run_meta.json", "stderr.txt"}
RUN_ID = re.compile(r"\d{8}T\d{6}(-\d+)?")
WRAPPER_PATH = re.compile(r"<repository root>\S*")
TEXT_SUFFIXES = {".txt", ".tsv", ".lg", ".md"}
JSON_SUFFIXES = {".json", ".jsonl"}


def command_key(meta: dict) -> tuple[str, ...]:
    """The recorded command without its interpreter (the first element when the second is a .py
    script) and with '/' separators, so the same command recorded on Windows and on macOS
    falls into one group."""
    cmd = [str(c).replace("\\", "/") for c in meta.get("command", [])]
    if len(cmd) >= 2 and cmd[1].endswith(".py"):
        cmd = cmd[1:]
    return tuple(cmd)


def records(question: Path) -> list[tuple[Path, dict]]:
    out = []
    for d in sorted(p for p in question.iterdir() if p.is_dir()):
        meta = d / "run_meta.json"
        if meta.is_file():
            out.append((d, json.loads(meta.read_text(encoding="utf-8"))))
    return out


def text_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    return [RUN_ID.sub("<run id>", WRAPPER_PATH.sub("<path>", l)) for l in lines]


def json_values(path: Path) -> list[object]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return [json.loads(text)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def compare_file(a: Path, b: Path) -> tuple[int, int, str]:
    """(units compared, units differing, unit name)."""
    if a.suffix == ".csv":
        with open(a, newline="", encoding="utf-8") as fa, open(b, newline="", encoding="utf-8") as fb:
            ra, rb = list(csv.reader(fa)), list(csv.reader(fb))
        n = sum(len(r) for r in ra)
        diff = abs(len(ra) - len(rb)) + sum(
            sum(x != y for x, y in zip(r1, r2)) + abs(len(r1) - len(r2)) for r1, r2 in zip(ra, rb))
        return n, diff, "cells"
    if a.suffix in JSON_SUFFIXES:
        va, vb = json_values(a), json_values(b)
        diff = abs(len(va) - len(vb)) + sum(x != y for x, y in zip(va, vb))
        return len(va), diff, "records"
    if a.suffix in TEXT_SUFFIXES:
        la, lb = text_lines(a), text_lines(b)
        diff = abs(len(la) - len(lb)) + sum(x != y for x, y in zip(la, lb))
        return len(la), diff, "lines"
    ba, bb = a.read_bytes(), b.read_bytes()
    diff = abs(len(ba) - len(bb)) + sum(x != y for x, y in zip(ba, bb))
    return len(ba), diff, "bytes"


def compare_question(question: Path) -> tuple[int, int]:
    """Return (pairs compared, pairs with a difference)."""
    groups: dict[tuple[str, ...], list[tuple[Path, dict]]] = {}
    for d, meta in records(question):
        groups.setdefault(command_key(meta), []).append((d, meta))
    pairs = bad_pairs = 0
    for key, recs in groups.items():
        ref, ref_meta = recs[0]
        if len(recs) < 2:
            print(f"[compare] {question.name}: only one record of `{' '.join(key)}` ({ref.name}); nothing to compare")
            continue
        for other, meta in recs[1:]:
            pairs += 1
            files = sorted(p for p in ref.iterdir() if p.is_file() and p.name not in SKIP)
            print(f"[compare] {question.name}: {ref.name} (host {ref_meta.get('host_id')}, "
                  f"{ref_meta.get('platform')}) vs {other.name} (host {meta.get('host_id')}, "
                  f"{meta.get('platform')}): {len(files)} files")
            differing = 0
            for f in files:
                g = other / f.name
                if not g.is_file():
                    print(f"[compare]   MISSING  {f.name}")
                    differing += 1
                    continue
                n, d, unit = compare_file(f, g)
                print(f"[compare]   {'SAME' if d == 0 else 'DIFFERS':8s} {f.name}: {d} of {n} {unit} differ")
                differing += d != 0
            bad_pairs += differing != 0
    return pairs, bad_pairs


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    pairs = bad = 0
    for q in argv:
        path = Path(q)
        if not path.is_dir():
            print(f"[compare] {q}: no such directory")
            return 2
        p, b = compare_question(path)
        pairs, bad = pairs + p, bad + b
    if pairs == 0:
        print("[compare] no pair of records was compared: this is not a pass")
        return 1
    print(f"[compare] {pairs - bad} of {pairs} record pairs identical")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
