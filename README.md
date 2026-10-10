# husm-aml-portsplit

Code and experiment pipeline for the paper *High-Utility Subgraph Mining for
Anti-Money Laundering: The Utility-Table Ceiling and Port-Splitting*.

The paper asks whether high-utility subgraph mining actually helps on transaction
graphs, and answers it in three parts:

1. **The utility-table ceiling.** With `R = e_max / e_min` the spread of the external
   utility table, any two patterns whose supports differ by more than a factor of `R`
   are ordered by support alone, whatever the utility values are. Under the Gaussian
   tables used in published evaluations `R` sits between 2 and 5, and the mined set
   overlaps plain frequent mining at Jaccard 0.825. Transaction amounts widen the table,
   so utility reverses the support order of many more pattern pairs than a Gaussian
   table does.
2. **Port-splitting.** Splitting each account into an in-port and an out-port lifts
   money-flow direction into vertex labels, so an undirected miner can tell gathering
   from scattering without any change to the engine. Typology recovery is then measured
   on the labelled laundering attempts of AMLworld.
3. **What tractability costs.** The transforms that make mining feasible on a
   million-vertex graph (degree cap, size cap, period cut) are measured against the
   signal they remove.

Everything reported in the paper is produced by the scripts below and is deterministic:
utility buckets are fixed quantiles, the miner is deterministic, and every learned
baseline runs under a fixed seed.

## Requirements

- Python 3.11 or newer, and the packages in `requirements.txt`
- JDK 17 or newer and Maven, for the mining engine (it is compiled for Java 17)
- 16 GB of RAM for the standard experiments; 64 GB for the full-graph runs

PyTorch and PyTorch Geometric are needed only by the graph-neural-network baselines;
every mining and recovery experiment runs without them.

## Quick start

```bash
python -m venv ~/.venvs/husm-aml-portsplit        # keep it outside any synced folder
source ~/.venvs/husm-aml-portsplit/bin/activate
pip install -r requirements.txt
mvn -f inchusm-miner/pom.xml package            # builds the engine and runs its tests
```

Then place the three public benchmarks under `datasets/` as described in
[docs/reproduce/paper3.md](docs/reproduce/paper3.md) and run the experiments from
`experiments/`.

## Which script produces which result

| Paper artifact | Script | Result file |
|---|---|---|
| Utility dimension vs frequent mining | `e8_utility_contrast.py` | `e8_utility_contrast.csv`, `e8_citeseer_seeds.csv` |
| Port-split recovery by degree cap | `e9_portsplit_recovery.py` | `e9_portsplit_recovery_*_raw.csv` |
| Recovery on the full graph | `e9_portsplit_recovery.py --mode full` | `e9_portsplit_recovery_HI-Small_full_raw.csv` |
| Bottleneck measure | `e9_flow_recovery.py` | `e9_flow_recovery_*.csv` |
| Recovery over eight typologies | `e3plus_recovery8.py` | `e3plus_recovery8_*.csv` |
| Detection under strict induction | `e4plus_gnn_drift.py` | `e4plus_gnn_drift.csv` |
| Account-level detection | `e7_actor_hus_detection.py` | `e7_actor_hus_detection.csv` |
| Figures | `make_figures_paper3.py` | PDF files, drawn only from result files |

`e9_portsplit_smoke.py` is a four-case check that the transform does what it claims:
without port-splitting, gathering, scattering and pass-through all collapse to one
single-edge pattern; with it, they separate. It needs no dataset and runs in seconds.

`run_all_paper3.py` runs everything in order and then grades the deterministic values
with `check_results.py`. A step can exit successfully and still produce wrong numbers,
so `check_results.py` is the real verdict.

## Recording runs and testing

Run every experiment through the recording wrapper, so that each result can be traced to
the code and data that produced it:

```bash
python experiments/provenance/run_and_record.py --into results-invariant/<question> \
    --data <input files> -- python experiments/scripts/<script>.py <arguments>
```

The scripts then write into the run's record directory, next to a `run_meta.json` holding
the command, commit, working-tree state, a hash of the sources, the engine jar's hash, the
data checksums, library versions and the machine; nothing earlier is overwritten. The
wrapper refuses to run when the engine jar is older than a Java source.
`experiments/provenance/machine_info.py --write` declares the measurement machine,
`verify_datasets.py` checks the benchmark files against `docs/reproduce/datasets.sha256`, and
`compare_records.py` compares records of one command made on different machines. Python tests:

```bash
python -m pytest experiments/tests
```

## Repository layout

```
inchusm-miner/                     Java mining engine (edu.huflit.inchusm)
  mining/                          HUSM.java static miner, upper bounds, utility tables
                                   IncHUSM.java incremental maintenance
  csp/                             subgraph matching; BottleneckSolver for the flow measure
  dfs/                             DFS codes and minimum-code canonical form
  graph/                           labelled graph and.lg loader
  cli/Main.java                    command line: mine, stream, match
  chain/                           conserved flow chain miner (directed transaction graphs)
    ChainMiner.java                level-wise miner, two-sided join, exact image sets
    ChainMain.java                 command line for the chain miner
  mutation/chain_mutations.py      fault injection: every injected fault must fail the chain tests
experiments/
  src/inchusm_aml/
    data/                          dataset loaders; portsplit.py holds the transform
    utility/monetary.py            monetary utility buckets, fit on train and frozen
    baselines/                     motif join, random forest gate, GNN, isomorphism
    eval/                          recovery and detection metrics
    miner_bridge.py                calls the Java engine over.lg files
  scripts/                         one entry point per experiment
  provenance/run_and_record.py     recording wrapper for every run
  tests/                           tests of the Python side
  results/                         earlier result tables (the small CSVs are committed)
results-invariant/                 recorded runs whose values do not depend on the machine
docs/reproduce/paper3.md           data sources, commands, reference numbers
datasets/                          not committed; see the reproduction guide
```

## Conserved flow chain miner

`edu.huflit.inchusm.chain` mines a directed transaction multigraph (accounts with labels,
transactions with a label and a positive amount) for flow chains: directed paths, and directed
cycles with a marked origin, whose occurrences keep the amount at every intermediate account,
`(1 - theta) a_in <= a_out <= (1 + theta) a_in`. The pair at the origin of a cycle (the money that
returns against the money that left) is not constrained. The support of a chain is the size of its
smallest image set over valid occurrences; the miner reports every chain up to a maximum length
whose support reaches a threshold, with exact image-set sizes.

Input format (`.tg`): `v ID LABEL` per account and `e SOURCE TARGET LABEL AMOUNT` per transaction.

```
mvn -f inchusm-miner/pom.xml package
java -cp inchusm-miner/target/inchusm-miner-0.1.0-SNAPSHOT.jar edu.huflit.inchusm.chain.ChainMain \
     --graph G.tg --theta 0.2 --min-support 10 --max-length 6 --out patterns.tsv
```

Tests: `ChainMinerTest` compares the miner with an independent enumerator of all occurrences
(`ChainBruteForce`) on random small graphs and checks hand-computed cases; `ChainTheoryTest`
checks, with the enumerator alone, that support never exceeds that of a chain's prefix or suffix
under the pairwise rule, and that the same check fails for a rule anchored on the first
transaction. `python inchusm-miner/mutation/chain_mutations.py` injects faults one at a time and
exits non-zero unless every one is caught.

## A note on scope

The engine also carries incremental and sliding-window mining, exercised by
`e5b_bhus_incremental.py`, `e5c_inchusm_mu.py` and `e6_sliding_window.py`. Those belong
to a separate line of work on incremental maintenance and are not used by any result in
this paper. They are kept because the static miner shares their code path.

## License

The source code is released under the MIT License; see `LICENSE`. Elliptic,
Elliptic++ and IBM AMLworld are public benchmarks distributed under their own terms and
are not part of this repository.
