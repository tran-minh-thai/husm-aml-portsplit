# experiments/scripts

One entry point per experiment. Each script adds `experiments/src` to `sys.path`,
writes its output to `experiments/results/`, and is deterministic where possible so the
numbers reproduce across machines.

Entry points (see docs/reproduce/paper3.md for commands and reference values):

| Script | Experiment |
|---|---|
| `e5a_ellipticpp_utility.py` | monetary-utility mining on the Elliptic++ actor graph |
| `e5b_bhus_incremental.py`   | BHUS-incremental representation, maintained per snapshot |
| `e5c_inchusm_mu.py`         | IncHUSM-mu (relative / pre-large thresholds) |
| `e3plus_recovery8.py`       | recovery over all eight AMLworld typologies, two scales |
| `e4plus_gnn_drift.py`       | detection vs strict-inductive GNN baselines, drift robustness |

The comparison baselines (motif join, random forest gate, subgraph isomorphism) live in
`inchusm_aml.baselines`. Every reported number is produced by running them here, under the
same protocol as the method being compared.
