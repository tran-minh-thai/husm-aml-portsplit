"""E4+ : detection vs strict-inductive GNN baselines, and drift robustness.

Reframed per the first paper's controlled null result (motif/RF features do NOT lift
transaction-level detection). This experiment does NOT claim to beat GNNs on detection.
It reports:
  (i)  the RF gate matches the strict-inductive GNN baselines under a clean protocol (the
       transductive advantage of GNNs is leakage, per Maganti 2026); and
  (ii) drift robustness: how detection degrades across the step-43 distribution shift, where
       every prior-learning classifier collapses, in contrast to the structure / typology
       layer (recovery in E3+, monetary utility in E5), which does not learn the label prior
       and therefore does not collapse.

The takeaway is consistent with the whole line of work: subgraph mining belongs at the
structural / typology layer, not at the transaction-scoring layer.

Build/obtain Elliptic under datasets/elliptic1 first (see docs/reproduce/paper3.md).
Run:  python scripts/e4plus_gnn_drift.py [--seeds 10] [--trees 100]
Output: results/e4plus_gnn_drift.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml.data.elliptic import load_elliptic  # noqa: E402
from inchusm_aml.baselines.rf_gate import run_rf_gate  # noqa: E402
from inchusm_aml.baselines import gnn_runner  # noqa: E402
from inchusm_aml.eval.metrics import drift_retention  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _row(method: str, res: dict) -> dict:
    return {
        "method": method,
        "f1_illicit": round(res["f1_mean"], 4),
        "f1_stable_35_42": round(res["zones"]["stable_35_42"], 4),
        "f1_collapse_43_49": round(res["zones"]["collapse_43_49"], 4),
        "drift_retention": round(drift_retention(res["zones"]), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--gnn", nargs="*", default=["gcn", "sage"],
                    help="GNN baselines to run (gcn, sage); EvolveGCN is cited, not run here")
    ap.add_argument("--gnn-epochs", type=int, default=200)
    ap.add_argument("--require-gnn", action="store_true",
                    help="fail instead of silently producing an RF-only table; use this when "
                         "the run is meant to back the paper's GNN rows")
    args = ap.parse_args()

    if args.require_gnn and not gnn_runner.available():
        raise SystemExit(
            "--require-gnn: torch_geometric is not importable, so the GNN rows cannot be "
            "produced. Install it (see docs/reproduce/paper3.md) or drop --require-gnn to "
            "regenerate the RF row alone.")

    # Record where the GNN rows were computed. A GNN F1 is not reproducible across devices
    # (torch.manual_seed does not make CUDA kernels deterministic), so "0.692" means little
    # without saying what produced it -- and this table already lost its provenance once.
    env = "RF only (torch_geometric absent)"
    if gnn_runner.available():
        import torch
        env = (f"torch {torch.__version__} on "
               f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  environment: {env}", flush=True)

    t0 = time.time()
    print("Loading Elliptic (strict-inductive: fit on steps <= 34)...", flush=True)
    data = load_elliptic()

    rows = [_row("RF (raw, 165-d)", run_rf_gate(data, seeds=tuple(range(args.seeds)),
                                                n_estimators=args.trees))]

    if gnn_runner.available():
        for model in args.gnn:
            print(f"  training strict-inductive {model}...", flush=True)
            try:
                rows.append(_row(model.upper(),
                                 gnn_runner.run_gnn_gate(data, model=model,
                                                         epochs=args.gnn_epochs)))
            except (NotImplementedError, ImportError) as exc:
                print(f"  [skip] {model}: {exc}", flush=True)
            except Exception as exc:  # a GNN failure must not lose the RF + drift result
                print(f"  [skip] {model} failed: {type(exc).__name__}: {exc}", flush=True)
    else:
        print("  torch_geometric absent: GNN baselines not run here. Under the SAME clean\n"
              "  inductive protocol the RF gate matches every GNN (Maganti 2026, Table);\n"
              "  their transductive advantage is leakage and is not comparable.")

    table = pd.DataFrame(rows)
    table["environment"] = env
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "e4plus_gnn_drift.csv"

    # Do NOT let a machine without torch_geometric quietly replace a table that HAS the GNN
    # rows with an RF-only one. That already happened once and it is silent: the run exits 0,
    # the RF number is still right, and the paper's GNN rows are left with no artifact behind
    # them. Refuse instead, and say what to do.
    if len(rows) == 1 and out.exists():
        try:
            prior = pd.read_csv(out)
        except Exception:
            prior = None
        if prior is not None and len(prior) > 1:
            raise SystemExit(
                f"refusing to overwrite {out.name}: it holds {len(prior)} rows "
                f"({', '.join(map(str, prior['method']))}) and this run produced RF alone, so "
                f"writing would destroy the only artifact behind the paper's GNN rows. Install "
                f"torch_geometric and re-run, or delete the file deliberately if the RF-only "
                f"table is what you want.")
    table.to_csv(out, index=False)

    print("\nDetection and drift robustness (illicit F1):")
    print(table.to_string(index=False))
    print("\nReading: detection scores are close under the clean protocol (no claim of beating\n"
          "GNNs); the low drift_retention shows every classifier collapses across the step-43\n"
          "shift. The structure/typology layer (E3+ recovery, E5 utility) does not learn the\n"
          "label prior and does not collapse -- that is where the value of mining sits.")
    print(f"\nDone ({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
