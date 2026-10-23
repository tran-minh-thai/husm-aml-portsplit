"""Map record commits that are not in this repository to commits with the same code.

A run on the measurement machine records the commit checked out there. When the machine pushes its results with
`git pull --rebase`, the local commits it made between runs are replaced by new ones, so a record can cite a commit
that the repository does not hold. Such a record is still tied to the code by its code_sha256 (run_and_record.py).

For every record under results/ and results-invariant/ whose git_commit is not in this repository, this tool
computes the code hash of the commits of the repository, with the function of run_and_record.py on the files of
each commit, and lists the commits whose hash equals the record's code_sha256. It first recomputes the hash of
records whose commit is present and refuses (exit 1) if any differs, so that a match is not an artefact of the
hashing.

    python experiments/provenance/commit_equivalence.py            # writes results/COMMIT_EQUIVALENCE.tsv

Refuses (exit 1) when a missing commit has no commit with the same code hash.
"""
from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "provenance"))
from run_and_record import CODE_ROOTS, code_hash  # noqa: E402

OUT = ROOT / "results" / "COMMIT_EQUIVALENCE.tsv"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def present(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", commit + "^{commit}"],
                          capture_output=True).returncode == 0


def hash_at(commit: str, cache: dict) -> str:
    if commit not in cache:
        with tempfile.TemporaryDirectory() as tmp:
            archive = pathlib.Path(tmp) / "tree.tar"
            # only the code that code_hash reads, as it existed at that commit
            roots = [b for b, _ in CODE_ROOTS if git("ls-tree", "--name-only", commit, "--", str(b)).strip()]
            subprocess.run(["git", "-C", str(ROOT), "archive", "-o", str(archive), commit, "--", *map(str, roots)],
                           check=True)
            with tarfile.open(archive) as tar:
                # the extraction filter exists from Python 3.11.4 and 3.12 on
                tar.extractall(tmp, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
            cache[commit] = code_hash(pathlib.Path(tmp))[0]
    return cache[commit]


def main() -> None:
    records = []
    for tree in ("results", "results-invariant"):
        for meta in sorted((ROOT / tree).glob("*/*/run_meta.json")):
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("git_commit") and m.get("code_sha256"):
                records.append((meta.parent.relative_to(ROOT).as_posix(), m["git_commit"], m["code_sha256"]))
    missing = [r for r in records if not present(r[1])]
    absent = {c for _, c, _ in missing}
    cache: dict = {}
    # the hashing is checked on records whose commit is present: their recomputed hash must equal the recorded one
    checked = {}
    for path, commit, sha in records:
        if commit not in absent and commit not in checked and len(checked) < 10:
            checked[commit] = (hash_at(commit, cache) == sha, path)
    bad = [p for ok, p in checked.values() if not ok]
    print(f"recomputed the code hash at {len(checked)} present commits: {len(checked) - len(bad)} equal to the record")
    if not checked or bad:
        sys.exit(f"commit_equivalence: the recomputed hash differs from the record for {bad}")
    if not missing:
        print("every record commit is in this repository")
        return
    wanted = {sha for _, _, sha in missing}
    commits = git("rev-list", "--reverse", "HEAD").split()
    matches = {sha: [c for c in commits if hash_at(c, cache) == sha] for sha in wanted}
    rows = []
    for path, commit, sha in missing:
        same = matches.get(sha, [])
        if not same:
            sys.exit(f"commit_equivalence: {path} cites {commit[:7]}; no commit of this repository has its code hash")
        rows.append({"record": path, "cited_commit": commit, "code_sha256": sha,
                     "equivalent_commit": same[0], "commits_with_same_code": len(same)})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} records cite commits not in this repository -> {OUT.relative_to(ROOT).as_posix()}")
    for r in rows:
        print(f"  {r['record']}: {r['cited_commit'][:7]} -> {r['equivalent_commit'][:7]} "
              f"(first of {r['commits_with_same_code']} commits with the same code)")


if __name__ == "__main__":
    main()
