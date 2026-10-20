"""What conservation changes when the chains are mined within time windows: open and closed chains, labeling S.

The transactions of one AMLworld data set are read once (same account codes and US Dollar rates as the full
chain input) and cut into consecutive windows of --window-days days from the first transaction; a last window
shorter than that is left out and reported. Every window is a graph of its own: its accounts are those of its
transactions (self-transactions dropped), its laundering accounts those with a laundering transaction in the
window, and its planted chains those of the laundering attempts whose transactions all lie in the window. In
every window the open chains are mined at theta = 0.2 and theta = inf and the closed chains at theta = 0.2 by
the miner; the closed chains at theta = inf are the accounts on a simple cycle (CycleMembership), as in
chain_cycle_effect.py --inf-engine cycles.

For every kind (open, closed) and length m the returned sets I_theta(m) of the windows are pooled as pairs
(window, account), and so are the marked accounts and the planted accounts: P, R, OR, and the retention ratios
r_L = |I_0.2 & L| / |I_inf & L| and r_B (likewise on the other accounts) of the pooled sets are written next to
the values of every window. One window covering the whole data set reproduces chain_conservation_effect.py
(open) and chain_cycle_effect.py (closed) on the full chain input.

Run: python experiments/scripts/windowed_effect.py --dataset HI-Large --window-days 16 [--tau 10] [--max-length 5]
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "probes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inchusm_aml.data.chain_input import write_tg  # noqa: E402
from inchusm_aml.eval import planted_chains as pc  # noqa: E402
from round2_measurements import load  # noqa: E402

JAR = ROOT / "inchusm-miner" / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
DAY = 1440


def odds(p: float) -> float:
    return p / (1 - p) if p < 1 else math.inf


def roles(path: Path, kind: str) -> dict[int, set]:
    r = pd.read_csv(path, sep="\t")
    r = r[r["kind"] == kind]
    if kind == "open":
        r = r[r["role"] == "intermediate"]
    return {int(m): set(g["account"].tolist()) for m, g in r.groupby("length")}


def mine(work: Path, theta: str, closed: bool, args) -> Path:
    out = work / f"accounts_{theta}_{'closed' if closed else 'open'}.tsv"
    cmd = ["java", f"-Xmx{args.heap}", "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain", "--graph", str(work / "g.tg"),
           "--theta", theta, "--min-support", str(args.tau), "--max-length", str(args.max_length), "--min-cycle", "3",
           "--out", str(work / "patterns.tsv"), "--accounts-out", str(out)] + ([] if closed else ["--no-closed"])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out


def cycles(work: Path, args) -> Path:
    out = work / "accounts_inf_closed.tsv"
    subprocess.run(["java", f"-Xmx{args.heap}", "-cp", str(JAR), "edu.huflit.inchusm.cycles.CycleMembership", "--graph",
                    str(work / "g.tg"), "--min-support", str(args.tau), "--min-length", "3", "--max-length",
                    str(args.max_length), "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    return out


def scores(i02: set, iinf: set, marked: set, planted: set) -> dict:
    row = {"planted": len(planted), "I_0.2": len(i02), "I_inf": len(iinf)}
    for t, s in (("0.2", i02), ("inf", iinf)):
        row[f"marked_{t}"] = len(s & marked)
        row[f"P_{t}"] = len(s & marked) / len(s) if s else float("nan")
        row[f"R_{t}"] = len(s & planted) / len(planted) if planted else float("nan")
    p1, p0 = row["P_0.2"], row["P_inf"]
    row["OR"] = odds(p1) / odds(p0) if i02 and iinf and 0 < p0 < 1 else float("nan")
    lm, bm = len(iinf & marked), len(iinf - marked)
    row["r_L"] = len(i02 & marked) / lm if lm else float("nan")
    row["r_B"] = len(i02 - marked) / bm if bm else float("nan")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="an AMLworld data set")
    ap.add_argument("--window-days", type=float, required=True, help="0: one window over the whole data set")
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=5)
    ap.add_argument("--heap", default="24g")
    args = ap.parse_args()

    edges, ptx, unit = load("amlworld", args.dataset)
    if unit != DAY:
        sys.exit(f"time unit {unit} per day, expected {DAY} (minutes)")
    flagged = edges["laundering"].to_numpy() == 1
    t = edges["time"].to_numpy()
    t0, t1 = int(t.min()), int(t.max())
    if args.window_days <= 0:                      # one window over the whole data set (the check against the full input)
        width, count, left = t1 - t0 + 1, 1, 0
    else:
        width = args.window_days * DAY
        count = int((t1 - t0 + 1) // width)
        left = (t1 - t0 + 1) - count * width
    print(f"{args.dataset}: {len(edges):,} transactions over {(t1 - t0 + 1) / DAY:.2f} days; {count} windows of "
          f"{args.window_days:g} days; {left / DAY:.2f} days at the end left out", flush=True)
    if count == 0:
        sys.exit("no complete window")
    attempt_span = ptx.groupby("attempt")["time"].agg(["min", "max"])
    pooled = {k: {m: [set(), set(), set(), set()] for m in range(2 if k == "open" else 3, args.max_length + 1)}
              for k in ("open", "closed")}
    rows = []
    for w in range(count):
        a, b = t0 + w * width, t0 + (w + 1) * width
        inside = (t >= a) & (t < b)
        e = edges[inside]
        laund_all = np.union1d(e["src"].to_numpy()[flagged[inside]], e["dst"].to_numpy()[flagged[inside]])
        e = e[e["src"].to_numpy() != e["dst"].to_numpy()]
        ids = np.union1d(e["src"].to_numpy(), e["dst"].to_numpy())
        marked = set(np.intersect1d(laund_all, ids).tolist())
        attempts = attempt_span.index[(attempt_span["min"] >= a) & (attempt_span["max"] < b)]
        planted = pc.planted_chains(ptx[ptx["attempt"].isin(attempts)], args.max_length)
        planted_open = {m: set(x for acc in g["accounts"] for x in acc[1:-1])
                        for m, g in planted[planted["kind"] == "open"].groupby("length")}
        planted_closed = {m: set(x for acc in g["accounts"] for x in acc)
                          for m, g in planted[planted["kind"] == "closed"].groupby("length")}
        with tempfile.TemporaryDirectory(prefix="windowed-") as tmp:
            work = Path(tmp)
            write_tg(pd.DataFrame({"id": ids, "label": 0}),
                     pd.DataFrame({"src": e["src"].to_numpy(), "dst": e["dst"].to_numpy(), "label": 0,
                                   "amount": e["amount"].to_numpy(dtype=float)}), work / "g.tg")
            at02 = mine(work, "0.2", True, args)
            sets = {("open", "0.2"): roles(at02, "open"), ("closed", "0.2"): roles(at02, "closed"),
                    ("open", "inf"): roles(mine(work, "inf", False, args), "open"),
                    ("closed", "inf"): roles(cycles(work, args), "closed")}
        for kind, plant in (("open", planted_open), ("closed", planted_closed)):
            for m, acc in pooled[kind].items():
                i02, iinf, c = sets[(kind, "0.2")].get(m, set()), sets[(kind, "inf")].get(m, set()), plant.get(m, set())
                rows.append({"window": w, "start_day": (a - t0) / DAY, "end_day": (b - t0) / DAY,
                             "transactions": int(len(e)), "accounts": int(len(ids)), "laundering_accounts": len(marked),
                             "attempts": int(len(attempts)), "kind": kind, "length": m, **scores(i02, iinf, marked, c)})
                for j, s in enumerate((i02, iinf, marked, c)):
                    acc[j] |= {(w, x) for x in s}
                print(f"window {w} {kind} {m}: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                      for k, v in rows[-1].items() if k in ("planted", "I_0.2", "I_inf", "P_0.2", "P_inf", "OR", "r_L", "r_B")),
                      flush=True)
    for kind in ("open", "closed"):
        for m, (i02, iinf, marked, c) in pooled[kind].items():
            rows.append({"window": "pooled", "start_day": 0.0, "end_day": count * width / DAY, "kind": kind,
                         "length": m, **scores(i02, iinf, marked, c)})
            print(f"pooled {kind} {m}: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                  for k, v in rows[-1].items() if k in ("planted", "I_0.2", "I_inf", "P_0.2", "P_inf", "R_0.2", "OR", "r_L", "r_B")),
                  flush=True)
    table = pd.DataFrame(rows)
    table.insert(0, "window_days", args.window_days)
    table.insert(0, "dataset", args.dataset)
    table.to_csv(OUT / "windowed_effect.csv", index=False)


if __name__ == "__main__":
    main()
