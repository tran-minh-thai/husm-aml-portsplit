"""Python <-> Java miner bridge (inchusm-miner/, Maven layout).

Language split: the miner is written in JAVA
(faithful to the papers  -  HUSM/DyFSM are both Java; fair runtime comparison with SPMF);
Python handles data/baseline/evaluation. The two sides communicate via FILE:
  Python exports snapshot -> .lg + utility table .tsv -> calls the JAR -> reads HUS .jsonl.
No JNI/py4j is used, keeping the two codebases separate.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JAR = (
    _PROJECT_ROOT / "inchusm-miner" / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
)


def export_lg(node_labels: pd.Series, edges: pd.DataFrame, path: Path) -> dict:
    """Write the labeled graph to .lg (GraMi/SPMF format).

    node_labels: Series index = original node id, value = int label (discretized
                 by the bucket table fit on the train window).
    edges: DataFrame with columns [src, dst] (+ optional 'elabel').
    Returns mapping original node id -> id 0..n-1 (to translate results back).
    """
    ids = {node: i for i, node in enumerate(node_labels.index)}
    lines = [f"v {ids[node]} {int(lab)}" for node, lab in node_labels.items()]
    has_elabel = "elabel" in edges.columns
    has_weight = "weight" in edges.columns          # FLOW measure: money on the edge (5th token)
    for r in edges.itertuples(index=False):
        el = int(getattr(r, "elabel")) if has_elabel else 0
        if has_weight:
            lines.append(f"e {ids[r.src]} {ids[r.dst]} {el} {float(getattr(r, 'weight')):.6g}")
        else:
            lines.append(f"e {ids[r.src]} {ids[r.dst]} {el}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return ids


def export_utilities(table: dict[int, float], path: Path) -> None:
    path.write_text(
        "\n".join(f"{int(k)} {v:.6f}" for k, v in sorted(table.items())) + "\n",
        encoding="ascii",
    )


def run_husm(
    graph_lg: Path,
    util_tsv: Path,
    min_util: float,
    mer: int = 1,
    bound: str = "ceu",
    max_edges: int = 5,  # project default: AML motif <= 5 vertices
    out_jsonl: Path | None = None,
    jar: Path = DEFAULT_JAR,
    timeout_s: int = 3600,
    heap: str = "8g",              # JVM max heap; the default JVM cap is 25% of RAM
    util_gauss: str | None = None,   # "MU,SIGMA,SEED": draw utilities as the HUSM papers do
    measure: str = "utility",        # "utility" | "support" | "flow" (u_flow = CMNI x bottleneck)
) -> tuple[pd.DataFrame, str]:
    """Run the Java miner, return (DataFrame of HUS, stdout summary).

    util_gauss replaces the table in util_tsv with utilities drawn from N(mu, sigma) per
    label, which is how the HUSM literature assigns external utility on most of its
    datasets; e8 uses it to contrast that protocol against real monetary utility.

    Build the jar first with: mvn -f inchusm-miner/pom.xml package
    """
    if not jar.exists():
        raise FileNotFoundError(
            f"Missing {jar}  -  run: mvn -f {jar.parents[1] / 'pom.xml'} package"
        )
    out_jsonl = out_jsonl or graph_lg.with_suffix(".hus.jsonl")
    util_arg = ["--util-gauss", util_gauss] if util_gauss else ["--util", str(util_tsv)]
    cmd = [
        "java", f"-Xmx{heap}", "-jar", str(jar), "mine", "--graph", str(graph_lg), *util_arg,
        "--minutil", str(min_util), "--mer", str(mer), "--bound", bound,
        "--max-edges", str(max_edges), "--measure", measure, "--out", str(out_jsonl), "--bhus",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"miner failed (exit {proc.returncode}):\n{proc.stderr}")
    rows = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows), proc.stdout


def run_inchusm_stream(
    snapshot_dir: Path,
    util_tsv: Path,
    min_value: float,
    measure: str = "utility",      # 'utility' | 'support'; flag present from the first version
    mer: int = 1,
    bound: str = "ceu",
    max_edges: int = 5,
    compare_static: bool = False,  # E1: correctness check + E2: measure speedup
    bhus_inc: bool = False,        # maintain BHUS boundaries per snapshot (e5b)
    mu_upper: float | None = None, # IncHUSM-mu relative large threshold (e5c)
    mu_lower: float | None = None, # IncHUSM-mu relative pre-large threshold (e5c)
    window: int = 0,               # sliding window length; 0 = add-only (e6)
    out_csv: Path | None = None,
    jar: Path = DEFAULT_JAR,
    timeout_s: int = 7200,
) -> tuple[pd.DataFrame, str]:
    """Run IncHUSM over the .lg snapshot sequence in snapshot_dir (ordered by file name).

    Return (per-snapshot telemetry DataFrame, stdout). When compare_static=True, the
    match='OK' column confirms INV2 (incremental high set == static re-run) at each
    snapshot  -  correctness (E1) + speedup (E2).
    """
    if not jar.exists():
        raise FileNotFoundError(
            f"Missing {jar}  -  run: mvn -f {jar.parents[1] / 'pom.xml'} package"
        )
    out_csv = out_csv or (snapshot_dir / "inchusm_stream.csv")
    cmd = [
        "java", "-jar", str(jar), "stream",
        "--snapshot-dir", str(snapshot_dir), "--util", str(util_tsv),
        "--minvalue", str(min_value), "--measure", measure, "--mer", str(mer),
        "--bound", bound, "--max-edges", str(max_edges), "--out", str(out_csv),
    ]
    if compare_static:
        cmd.append("--compare-static")
    if bhus_inc:
        cmd.append("--bhus-inc")
    if mu_upper is not None:
        cmd += ["--mu-upper", str(mu_upper)]
        if mu_lower is not None:
            cmd += ["--mu-lower", str(mu_lower)]
    if window > 0:
        cmd += ["--window", str(window)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"IncHUSM failed (exit {proc.returncode}):\n{proc.stderr}")
    df = pd.read_csv(out_csv)
    if compare_static and (df["match"] == "MISMATCH").any():
        bad = df[df["match"] == "MISMATCH"]["snapshot"].tolist()
        raise AssertionError(f"INV2 BROKEN at snapshot {bad}  -  incremental != static re-run")
    return df, proc.stdout


def export_snapshot_stream(snapshots, out_dir: Path) -> list[dict]:
    """Write each snapshot (node_labels: Series, edges: DataFrame) to snap_XX.lg.

    Used for E1/E2: convert the 49 Elliptic time-steps (labels already discretized) into
    a .lg sequence for the `stream` command. Return the id-mapping for each snapshot.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mappings = []
    for i, (labels, edges) in enumerate(snapshots, start=1):
        mappings.append(export_lg(labels, edges, out_dir / f"snap_{i:02d}.lg"))
    return mappings
