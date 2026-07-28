# inchusm-miner

Faithful **Java re-implementation of HUSM** - *High Utility Subgraph Mining in single
graph databases* (Chen, He, Chen, Gan, Fournier-Viger, Information Sciences 675 (2024)
120743, DOI 10.1016/j.ins.2024.120743) - and the base for its incremental extension
**IncHUSM** (high-utility subgraph mining over a stream of disjoint transaction
snapshots, targeting anti-money-laundering on blockchain data).

The original authors' code is not public. This implementation follows the paper's
definitions and algorithms directly:

| Paper | Here |
|---|---|
| Minimum DFS code / canonical form (gSpan; HUSM Sec.4.1, Lemma 1) | `dfs/DFSCode`, `dfs/MinCode` |
| MER constraint + CMNI table via CSP (Defs. 6-7, 10) | `csp/CSPSolver`, `csp/CMNITable`, `csp/BallCache` |
| Utility u(P) = CMNI x mean label utility (Def. 8) | `mining/ValueCompute` |
| Upper bounds IVU / EU / CEU (Defs. 11-14, Lemma 3) | `mining/ValueCompute` |
| HUSM Algorithms 1-3 | `mining/HUSM` |
| Boundary HUS concise representation (Def. 15, Alg. 5) | `mining/BHUS` |
| **IncHUSM** - incremental mining over disjoint snapshots (Lemmas L1-L3) | `mining/IncHUSM`, `mining/PatternState`, `mining/Segment` |
| Measure flag utility / support (build-time flag) | `mining/Measure` |

## Build & run

```bash
mvn package                       # runs the full test suite, produces target/inchusm-miner-*.jar

# static HUSM on one graph:
java -jar target/inchusm-miner-0.1.0-SNAPSHOT.jar mine \
     --graph citeseer.lg --util-gauss 3.0,1.0,42 \
     --minutil 50 --mer 1 --bound ceu --out hus.jsonl --bhus

# incremental IncHUSM over a stream of disjoint snapshots (a.lg per time step):
java -jar target/inchusm-miner-0.1.0-SNAPSHOT.jar stream \
     --snapshot-dir snapshots/ --util u.tsv --minvalue 50 \
     --measure utility --mer 1 --bound ceu --max-edges 5 \
     --compare-static --out per_snapshot.csv
```

- Graph input: `.lg` format (GraMi/SPMF): `v ID LABEL` / `e U V [LABEL]`.
- Utilities: `--util FILE` (`LABEL VALUE` per line) or `--util-gauss MU,SIGMA,SEED`
  (the paper's N(3,1) convention for datasets without native utilities; deterministic per seed).
- `--bound ivu|eu|ceu` selects the pruning bound (ablation); the result set is identical
  by construction, only candidate counts/runtime change.
- `--max-edges K` caps pattern size (the AML use case mines <=5-node motifs).
- `stream --measure support` reduces IncHUSM to incremental frequent-subgraph mining
  (CMNI is its own anti-monotone bound) - the ablation that isolates utility's value.
- `stream --compare-static` re-runs static HUSM on the accumulated union after each
  snapshot and reports `OK`/`MISMATCH` plus the incremental-vs-rerun speedup.

## IncHUSM invariants (checked in tests against static re-run)

- **INV1** every stored pattern's u / CEU equals static HUSM on the union graph - 
  `PatternState` sums per-snapshot segments (disjoint snapshots => additive, Lemma L1).
- **INV2** the reported high set after snapshot *t* equals static HUSM on G1..t.
- **INV3** with an absolute threshold, value and bound are monotone in *t*, so a
  pattern only moves candidate -> high (append-only alert stream).

Guided DFS touches only the arriving snapshot: a pattern absent from Gt whose subtree
was already expanded is frozen and skipped (Lemma L3); new patterns are back-filled
over their parent's occurrence snapshots only.

## Correctness

`mvn test` (12 tests) verifies, among others:

- **Static brute-force equivalence**: HUSM's result set equals a definition-level
  reference (independent occurrence enumeration + BFS MER filter), for *every* bound
  and multiple thresholds (`HUSMEquivalenceTest`).
- **Incremental = static after every snapshot**: streaming disjoint snapshots through
  IncHUSM matches brute-force on the accumulated union at each step, for both the
  utility and support measures, and cross-checks against `HUSM.mine`; high set grows
  monotonically (`IncHUSMEquivalenceTest`, INV1-INV3).
- **Canonical-form invariance**: minimum DFS code is invariant under random vertex
  permutations (`MinCodeTest`).
- **Bound dominance**: u <= CEU <= EU <= IVU on every evaluation, and candidate counts
  ordered IVU >= EU >= CEU.

Note on reproducing the HUSM paper's Citeseer pattern counts: its authors drew
external utilities from N(3,1) with an unpublished seed, so exact counts are not
reproducible - only trends (counts decrease with minUtil; candidates ordered
IVU >= EU >= CEU) can be checked.

## How the two sides fit together

The Java engine does the mining. The Python side under `../experiments/` prepares the
graphs, derives the utility tables, and evaluates the results. They communicate through
files: `.lg` for the labelled graph, `.tsv` for the utility table, and `.jsonl` for the
mined patterns. `experiments/src/inchusm_aml/miner_bridge.py` is the only place that
calls the jar, so the interface is in one file.
