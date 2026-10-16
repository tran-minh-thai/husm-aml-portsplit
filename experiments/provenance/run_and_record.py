#!/usr/bin/env python3
"""Run a command and record it, so that a number can be traced back later.

A run whose output was only read on screen is a run that no longer exists: the
number may survive in a document, but what produced it does not. This wrapper
makes recording the default rather than something to remember -- run everything
through it, including the quick check you did not plan to cite, because that is
the run whose number gets quoted later.

What every record carries: the command verbatim, the repository root, the git
commit and whether the working tree was clean at the time, the platform and
language versions, the output, and the exit code.

Two things this tool refuses to do quietly:

* run outside a git repository. It used to record git's own error message in
  the commit field and call the tree "clean", because an empty status listing
  looks like a clean tree. A record that names a commit that never existed is
  worse than no record: it stops people asking. Now it exits non-zero and
  writes nothing.
* keep machine-dependent timing in a machine-independent tree (timing of a probe goes to the paired
  tree results-probe-timing/, whose records say they are probes and which no paper tool reads). Output written
  under results-invariant/ or results-probe/ has its timing lines redacted and
  no finish time recorded, so nothing in those trees can be read as a
  performance number. The number of redacted lines is recorded so the
  redaction itself is visible.

Two things specific to this repository:

* the mining engine runs from a jar the Python drivers call. A jar older than any Java source
  may not be the code at the recorded commit, so the run is refused until the jar is rebuilt
  (mvn -f inchusm-miner/pom.xml package). The jar's sha256 is recorded.
* the record carries a hash of every source file (Python drivers and Java engine, line endings
  normalized so one commit gives one hash on every operating system) and the sha256 and size
  of every data file named with --data.

And one thing it no longer writes at all: an absolute path of the machine the
run was made on. Records used to carry the repository root and working
directory as absolute paths, and Maven's own output repeats the repository
root. That path names a user's home directory, means nothing on any other
machine, and is exactly what a published artifact must not carry. Paths are
now recorded relative to the repository root, and every occurrence of the
repository root or the home directory in the captured output is replaced by a
placeholder. The number of replacements is recorded, like the timing
redaction, so the scrubbing is visible rather than silent.

A run that produces both kinds of number writes them into two records with
the same run id: the machine-independent one under --into, and, with
--timing-into, a paired machine-dependent one. The command learns both
directories from the environment (RUN_RECORD_DIR, RUN_TIMING_DIR, RUN_TIMING_KIND, RUN_ID),
which exist before it starts. A paired record under results/ is accepted only
on the declared measurement machine: results/MEASUREMENT_MACHINE.txt must hold
the host id this tool records (a short hash of the host name, so the name
itself is never published).

Usage:
    python experiments/provenance/run_and_record.py --into results-invariant/currency \
        --data datasets/amlworld/HI-Small_Trans.csv -- python experiments/probes/x.py
    python experiments/provenance/run_and_record.py --into results-invariant/q --timing-into results/q -- ...
    python experiments/provenance/run_and_record.py --print-host-id
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import re
import socket
import subprocess
import tempfile
import sys

MACHINE_INDEPENDENT_TREES = {"results-invariant", "results-probe"}
MEASUREMENT_TREE = "results"
PROBE_TIMING_TREE = "results-probe-timing"
MEASUREMENT_MACHINE_FILE = "MEASUREMENT_MACHINE.txt"
TIMING_LINE = re.compile(r"Time elapsed|Total time|Finished at|elapsed|^Ran \d+ tests? in ", re.IGNORECASE)
REDACTED = "[timing redacted: machine-dependent, not kept in this tree]"
ROOT_PLACEHOLDER = "<repository root>"
HOME_PLACEHOLDER = "<home>"
TEMP_PLACEHOLDER = "<temporary directory>"


def first_line(args, cwd=None):
    try:
        result = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
        text = (result.stdout or result.stderr).strip()
        return text.splitlines()[0] if text else "unavailable"
    except Exception:
        return "unavailable"


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(on, kernel32=None):
    """On Windows, ask the system not to sleep while a run is in progress (on=True) and release it (on=False).

    A run of several hours is lost without an error when the machine sleeps. Returns the state flags requested,
    or None where there is nothing to do (other systems).
    """
    if kernel32 is None:
        if os.name != "nt":
            return None
        import ctypes
        kernel32 = ctypes.windll.kernel32
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS
    kernel32.SetThreadExecutionState(flags)
    return flags


def repository_root():
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return pathlib.Path(result.stdout.strip())


def redact_timing(text):
    kept = []
    redacted = 0
    for line in text.splitlines():
        if TIMING_LINE.search(line):
            kept.append(REDACTED)
            redacted += 1
        else:
            kept.append(line)
    return "\n".join(kept) + ("\n" if text.endswith("\n") else ""), redacted


RESULT_TREES = ("results", "results-invariant", "results-probe", "results-probe-timing")


def tree_state(root):
    """'clean'; 'clean (N untracked result records)' when the only changes are new files under a result
    tree -- records of earlier runs not yet committed, as in a campaign; or 'MODIFIED (N files)'.

    Only the first two describe the code that ran. Untracked lines are listed one per file so a
    record directory counts by its files.
    """
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], capture_output=True,
                            text=True, cwd=root)
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    results = [line for line in lines
               if line.startswith("?? ") and line[3:].strip('"').split("/", 1)[0] in RESULT_TREES]
    others = len(lines) - len(results)
    if others:
        return f"MODIFIED ({others} files)"
    return "clean" if not results else f"clean ({len(results)} untracked result records)"


def scrub_paths(text, root):
    """Replace the repository root, the system temporary directory, then the home directory, by placeholders.

    The root goes first because it lies inside the home directory: the other
    order would leave '<home>/...' fragments that still name the layout of
    this machine. Returns the scrubbed text and the number of replacements.
    """
    home = str(pathlib.Path.home())
    base = tempfile.gettempdir()
    temporary = sorted({base, os.path.realpath(base)} - {"/", ""}, key=len, reverse=True)
    count = 0
    pairs = [(str(root), ROOT_PLACEHOLDER)] + [(t, TEMP_PLACEHOLDER) for t in temporary] + [(home, HOME_PLACEHOLDER)]
    for absolute, placeholder in pairs:
        count += text.count(absolute)
        text = text.replace(absolute, placeholder)
    return text, count


JAR = pathlib.Path("inchusm-miner") / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
JAVA_SOURCES = pathlib.Path("inchusm-miner") / "src" / "main" / "java"
CODE_ROOTS = (("experiments", ".py"), (str(JAVA_SOURCES), ".java"), ("inchusm-miner/pom.xml", ""))


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def jar_state(root):
    """(sha256 of the engine jar or None if absent, Java sources newer than the jar)."""
    jar = root / JAR
    if not jar.is_file():
        return None, []
    built = jar.stat().st_mtime
    stale = sorted(source.relative_to(root).as_posix() for source in (root / JAVA_SOURCES).rglob("*.java")
                   if source.stat().st_mtime > built)
    return sha256_file(jar), stale


def code_hash(root):
    """sha256 over every source file (path with '/' separators, then content with CRLF -> LF), in
    path order, so one commit gives one hash on macOS, Linux and Windows. Returns (hash, file count)."""
    files = []
    for base, suffix in CODE_ROOTS:
        path = root / base
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(f for f in path.rglob("*" + suffix)
                         if f.is_file() and "__pycache__" not in f.parts)
    digest = hashlib.sha256()
    for f in sorted(files, key=lambda f: f.relative_to(root).as_posix()):
        digest.update(f.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(f.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def data_record(paths, root):
    """sha256 and size of every data file, named relative to the repository root."""
    out = []
    for p in paths:
        path = pathlib.Path(p).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--data {p}: no such file (resolved to {relative_to_root(path, root)})")
        out.append({"path": relative_to_root(path, root), "bytes": path.stat().st_size,
                    "sha256": sha256_file(path)})
    return out


def library_versions():
    versions = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "networkx", "matplotlib", "torch"):
        try:
            versions[name] = getattr(__import__(name), "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def host_id():
    """A short hash of the host name: tells machines apart without publishing the name."""
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]


def machine_description():
    """CPU model and memory, read from the operating system; 'unavailable' when it cannot be read."""
    system = platform.system()
    if system == "Darwin":
        cpu = first_line(["sysctl", "-n", "machdep.cpu.brand_string"])
        memory = first_line(["sysctl", "-n", "hw.memsize"])
    elif system == "Linux":
        cpu = first_line(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"])
        memory = first_line(["sh", "-c", "grep -m1 MemTotal /proc/meminfo"])
    elif system == "Windows":
        cpu = platform.processor() or "unavailable"
        memory = "unavailable"
    else:
        cpu, memory = "unavailable", "unavailable"
    return {"cpu": cpu.strip(), "memory": memory.strip(), "logical_cpus": os.cpu_count()}


def measurement_tree_refusal(timing_parent, root):
    """Why a paired record may not go under results/ on this machine, or None when it may."""
    try:
        relative = timing_parent.resolve().relative_to(root.resolve())
    except ValueError:
        return None  # outside the repository: a probe kept elsewhere, not a measurement
    if not relative.parts or relative.parts[0] != MEASUREMENT_TREE:
        return None
    declared = root / MEASUREMENT_TREE / MEASUREMENT_MACHINE_FILE
    if not declared.is_file():
        return (f"{declared.relative_to(root).as_posix()} does not exist; declare the measurement machine by writing "
                f"its host id there (this machine's is {host_id()})")
    ids = [line.strip() for line in declared.read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    if not ids:
        return (f"{declared.relative_to(root).as_posix()} exists but declares no host id "
                f"(this machine's is {host_id()})")
    if host_id() not in ids:
        return (f"this machine (host id {host_id()}) is not the declared measurement machine; "
                f"{declared.relative_to(root).as_posix()} declares {ids}")
    return None


def timing_kind(timing_parent, root):
    """'measurement' under results/, 'probe' under results-probe-timing/, else 'outside the result trees'."""
    try:
        first = timing_parent.resolve().relative_to(root.resolve()).parts[:1]
    except ValueError:
        return "outside the result trees"
    if first == (MEASUREMENT_TREE,):
        return "measurement"
    if first == (PROBE_TIMING_TREE,):
        return "probe: development machine, decides whether a change is worth making, never cited"
    return "outside the result trees"


def probe_pairing_refusal(target_parent, timing_parent, root):
    """Probe timing pairs only with a probe record, so a probe number cannot sit next to a measurement."""
    if timing_kind(timing_parent, root).startswith("probe"):
        try:
            first = target_parent.resolve().relative_to(root.resolve()).parts[:1]
        except ValueError:
            first = ()
        if first != ("results-probe",):
            return f"--timing-into {PROBE_TIMING_TREE}/ pairs only with --into results-probe/"
    return None


def new_record_dirs(parents, stamp):
    """parent/run_id for every parent, with one run id free under all of them: stamp, or stamp-N.

    Created here and never written into twice. Runs started in the same second (in one checkout or
    in different checkouts sharing a timing directory) race for an id, so the id is taken by creating
    its directories, which fails if one exists already: on a collision the directories this call made
    for that id are removed and the next suffix is tried. Checking first and creating after would let
    two runs pick the same id and the slower one fail.
    """
    suffix = 1
    while True:
        run_id = stamp if suffix == 1 else f"{stamp}-{suffix}"
        created = []
        try:
            for parent in parents:
                parent.mkdir(parents=True, exist_ok=True)
                target = parent / run_id
                target.mkdir()
                created.append(target)
            return created
        except FileExistsError:
            for target in created:
                target.rmdir()
            suffix += 1


def relative_to_root(path, root):
    try:
        return pathlib.Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return "(outside the repository)"


def main():
    if sys.argv[1:] == ["--print-host-id"]:
        print(host_id())
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--into", required=True, help="directory the run record is written under")
    parser.add_argument("--timing-into", default=None,
                        help="directory a paired machine-dependent record is written under (same run id)")
    parser.add_argument("--note", default="", help="what question this run answers")
    parser.add_argument("--data", nargs="*", default=[], help="input data files to checksum")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        print("no command given; put it after --", file=sys.stderr)
        return 2

    root = repository_root()
    if root is None:
        print(f"refusing to record: {pathlib.Path.cwd()} is not inside a git repository, "
              "so no commit could be attached to this run", file=sys.stderr)
        return 2

    target_parent = pathlib.Path(args.into).resolve()
    machine_independent = bool(MACHINE_INDEPENDENT_TREES.intersection(target_parent.parts))
    timing_parent = pathlib.Path(args.timing_into).resolve() if args.timing_into else None
    if timing_parent is not None:
        if MACHINE_INDEPENDENT_TREES.intersection(timing_parent.parts):
            print(f"refusing to run: --timing-into {args.timing_into} lies in a machine-independent tree, "
                  f"where no timing may be kept", file=sys.stderr)
            return 2
        refusal = measurement_tree_refusal(timing_parent, root) or probe_pairing_refusal(target_parent, timing_parent, root)
        if refusal:
            print(f"refusing to run: {refusal}", file=sys.stderr)
            return 2

    # The tree state that describes the code which ran is the state BEFORE the run. A command
    # may legitimately write into the tree (a tool writing its own report), and reading the
    # state afterwards would then call a clean run modified. The state after the run is kept
    # too, when it differs, so a command that altered tracked sources is still visible.
    jar_sha, stale = jar_state(root)
    if stale:
        print(f"refusing to run: {len(stale)} Java source file(s) are newer than {JAR.as_posix()}, so the engine "
              f"that would run may not be the sources at this commit; rebuild first "
              f"(mvn -f inchusm-miner/pom.xml package). First: {stale[:3]}", file=sys.stderr)
        return 2
    try:
        data = data_record(args.data, root)
    except FileNotFoundError as e:
        print(f"refusing to run: {e}", file=sys.stderr)
        return 2
    code_sha, code_files = code_hash(root)
    tree_before = tree_state(root)
    commit = first_line(["git", "rev-parse", "HEAD"], cwd=root)
    started = datetime.datetime.now()
    # The record directories exist before the command starts, so it can write its own files into
    # them. Two runs started in the same second used to share one directory, the second overwriting
    # the first without a word; a second run in the same second now gets a suffix.
    parents = [target_parent] + ([timing_parent] if timing_parent is not None else [])
    created = new_record_dirs(parents, started.strftime("%Y%m%dT%H%M%S"))
    target = created[0]
    stamp = target.name
    timing_target = created[1] if timing_parent is not None else None
    child_env = dict(os.environ, RUN_RECORD_DIR=str(target), RUN_ID=stamp)
    if timing_target is not None:
        child_env["RUN_TIMING_DIR"] = str(timing_target)
        child_env["RUN_TIMING_KIND"] = timing_kind(timing_parent, root)
    keep_awake(True)
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=child_env)
    finally:
        keep_awake(False)
    finished = datetime.datetime.now()
    tree_after = tree_state(root)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        print(f"refusing to record: could not read HEAD of {root} ({commit})", file=sys.stderr)
        return 2

    stdout, stderr = result.stdout, result.stderr
    redacted = 0
    if machine_independent:
        stdout, n1 = redact_timing(stdout)
        stderr, n2 = redact_timing(stderr)
        redacted = n1 + n2
    stdout, p1 = scrub_paths(stdout, root)
    stderr, p2 = scrub_paths(stderr, root)
    recorded_command = []
    p3 = 0
    for argument in command:
        scrubbed, n = scrub_paths(argument, root)
        recorded_command.append(scrubbed)
        p3 += n
    scrubbed_paths = p1 + p2 + p3

    (target / "stdout.txt").write_text(stdout, encoding="utf-8")
    (target / "stderr.txt").write_text(stderr, encoding="utf-8")

    meta = {
        "run_id": stamp,
        "note": args.note,
        "command": recorded_command,
        "repository": root.name,
        "working_directory": relative_to_root(pathlib.Path.cwd(), root),
        "tree_kind": "machine-independent" if machine_independent else "machine-dependent",
        "timing_lines_redacted": redacted,
        "absolute_paths_scrubbed": scrubbed_paths,
        "exit_code": result.returncode,
        "started_at": started.isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_tree": tree_before,
        "code_sha256": code_sha,
        "code_files": code_files,
        "engine_jar_sha256": jar_sha,
        "data": data,
        "libraries": library_versions(),
        "host_id": host_id(),
        "machine_description": machine_description(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "java": first_line(["java", "-version"]),
        "maven": first_line(["mvn", "-v"]),
    }
    if tree_after != tree_before:
        meta["git_tree_after_run"] = tree_after
    if not machine_independent:
        meta["finished_at"] = finished.isoformat(timespec="seconds")
    (target / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if timing_target is not None:
        paired = dict(meta, tree_kind="machine-dependent", timing_kind=timing_kind(timing_parent, root),
                      timing_lines_redacted=0,
                      finished_at=finished.isoformat(timespec="seconds"),
                      paired_with=relative_to_root(target, root))
        (timing_target / "run_meta.json").write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
        meta_note = relative_to_root(timing_target, root)
        meta["paired_timing_record"] = meta_note if meta_note != "(outside the repository)" else "(outside the repository)"
        (target / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    print(f"\nrecorded: {target}" + (f"  ({redacted} timing lines redacted)" if redacted else "")
          + (f"  ({scrubbed_paths} absolute paths scrubbed)" if scrubbed_paths else ""))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
