# Reproducing the results

Every number comes from a script in `experiments/scripts/`. Run through the recording
wrapper (section 3), a script writes into its own record directory under
`results-invariant/<question>/<run id>/` together with `run_meta.json` (command, commit,
working-tree state, source hash, engine-jar hash, data checksums, library versions,
machine), and never overwrites an earlier result. The CSVs under `experiments/results/`
are the earlier results and are kept unchanged. This guide covers the environment, the
data, the command behind each result, and the values to expect.

## 1. Environment

```bash
python -m venv ~/.venvs/husm-aml-portsplit        # Windows: %USERPROFILE%\.venvs\husm-aml-portsplit
source ~/.venvs/husm-aml-portsplit/bin/activate   # Windows:...\Scripts\activate
pip install -r requirements.txt
mvn -f inchusm-miner/pom.xml package            # JDK 17 or newer; builds the engine and runs its tests
```

The build must finish with all tests passing before any experiment is run; the Python
side calls the resulting jar through `miner_bridge.py`.

Add PyTorch and PyTorch Geometric only for the graph-neural-network baselines. The CPU
wheels are enough and are far smaller than the CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

## 2. Data

Place the three benchmarks under `datasets/`, which is not tracked by git.

| Path | Used by |
|---|---|
| `datasets/elliptic1/elliptic_txs_{features,classes,edgelist}.csv` | detection under strict induction |
| `datasets/ellipticpp/{AddrAddr_edgelist,wallets_features,wallets_classes}.csv` | monetary utility, account-level detection |
| `datasets/amlworld/{HI,LI}-{Small,Medium}_Trans.csv` and `_Patterns.txt` | typology recovery, port-splitting |
| `datasets/citeseer/citeseer.lg` | the utility-dimension contrast |

Every file is listed with its size and sha256 in `docs/reproduce/datasets.sha256`. Check a
copy before running anything:

```bash
python experiments/provenance/verify_datasets.py      # must end: all 15 files match the manifest
```

Elliptic and Elliptic++ come from the Elliptic data set release; AMLworld is the IBM
synthetic anti-money-laundering benchmark. Both are distributed on Kaggle under their
own terms. Citeseer is the standard citation graph, converted to the engine's `.lg`
format (one `v id label` line per vertex, one `e src dst label` line per edge).

Transaction counts in the paper are taken after self-transfers are dropped, so the raw
file is larger than the reported figure: HI-Small holds 5,078,345 rows and 4,487,133
transactions between distinct accounts.

## 3. Running everything

```bash
python experiments/scripts/run_all_paper3.py          # any OS
run_paper3.bat                                        # Windows convenience wrapper
```

To record a single run (recommended for every run, including quick checks):

```bash
python experiments/provenance/run_and_record.py --into results-invariant/<question> \
    --data <input files> -- python experiments/scripts/<script>.py <arguments>
```

The wrapper refuses to run when the engine jar is older than any Java source. The tests
of the Python side run with `python -m pytest experiments/tests`.

The measurement machine is declared once, by running
`python experiments/provenance/machine_info.py --write` on it and committing
`results/MEASUREMENT_MACHINE.txt` (host id, CPU, cores, RAM, operating system, Python and its
libraries, Java runtime version and vendor, Maven). Timing records under `results/` are
refused on any other machine. Records of the same command made on different machines are
compared exactly, cell by cell, with
`python experiments/provenance/compare_records.py results-invariant/<question>`.

`run_all_paper3.py` runs each step in order, times it, continues past a failure, prints a
summary, and finally grades the deterministic values with `check_results.py`. A step can exit
successfully and still produce wrong numbers, so `check_results.py` is the real verdict:
it checks only values that must reproduce on any machine and never checks timing.

A full run needs roughly 16 GB of RAM. The full-graph variants of the port-split
experiment need 64 GB and take one to two hours each.

## 4. Running one experiment at a time

All commands are given from `experiments/`.

### The utility dimension against frequent mining

Mines Citeseer twice at the same threshold: once with utilities drawn from `N(3, 1)`,
the protocol published with the method, and once with a constant table. A constant table
makes utility a fixed multiple of support, so the second run is plain frequent subgraph
mining sharing the same engine and the same threshold. The overlap between the two
answers measures how much the utility dimension contributed.

```bash
python scripts/e8_utility_contrast.py                 # about 1 hour
python scripts/e8_utility_contrast.py --skip-mine     # seconds; re-scoring only
```

The utility seed of the original protocol was never published, so the script sweeps ten
seeds and reports the median. Per-seed rows land in `e8_citeseer_seeds.csv`. Do not
quote seed 42 alone: it gives the smallest share of reversed pairs of the ten and
overstates the effect.

Over the pattern pairs with different support:

| Graph and table | R | locked pairs | reversed pairs | utility share | Jaccard vs frequent |
|---|---|---|---|---|---|
| Citeseer, `N(3, 1)`, 6 labels (median of 10 seeds) | 2.26 | 52.7% | 9.0% | 3.1% | 0.825 |
| Elliptic++, `N(3, 1)`, 4 labels | 2.19 | 70.2% | 3.9% | 2.4% | - |
| Elliptic++, monetary, 4 buckets | 4456 | 0.0% | 14.8% | 2.2% | - |

"Locked pairs" are those whose support ratio exceeds R, so no utility table of that
spread can change their order (the mean utility is trapped between the smallest and
largest table entries). "Reversed pairs" are those the utility score actually orders the
other way round from support; every reversed pair is unlocked, and the script stops if a
locked pair is reversed. "Utility share" is the Shapley share of Var(log u) carried by the
utility term, (Var(log ubar) + Cov(log CMNI, log ubar)) / Var(log u); it is not a measure
of reordering: on Elliptic++ the monetary table reverses almost four times as many pairs
as the Gaussian one while both shares are near 2%, because log ubar and log CMNI are
strongly negatively correlated under the monetary table.

### Port-splitting and typology recovery

Splits every account into an in-port and an out-port, mines the transformed graph, and
checks whether each mined pattern occurs inside the port-split subgraph of a labelled
laundering attempt, covering at least three distinct accounts.

```bash
python scripts/e9_portsplit_smoke.py                              # transform check, no data
python scripts/e9_portsplit_recovery.py --dataset HI-Small \
    --mode truth-neighborhood --utility-scale raw \
    --min-util 200 --max-edges 3 --hub-caps 0 200 50
python scripts/e9_portsplit_recovery.py --dataset HI-Small \
    --mode full --utility-scale raw --min-util 12000 --heap 32g   # needs 64 GB
```

`--min-util` is absolute while pattern utility grows with graph size, so it must be
scaled with the graph: 12,000 on the full graph corresponds to 200 on the
truth-neighbourhood subgraph, which is 57 times smaller. The driver warns when the
threshold is out of scale for the graph it was given.

`--utility-scale raw` uses transaction amounts directly. The `log` setting narrows the
table spread and with it the share of pattern pairs utility can reorder. The driver
prints the spread, the locked and reversed pair shares and the variance decomposition on
every run, stops if a locked pair is reversed, and warns if utility reverses no pair.

AMLworld amounts are converted to US Dollar when loaded (see
`experiments/src/inchusm_aml/data/amlworld.py`); the recovery tables reported for the
submitted manuscript (commit `aed079e`) were computed from amounts summed across
currencies and are superseded. Reference values for this section will be listed once
regenerated.

A bipartite core needs more than three edges to be told apart from a fan, and
`--max-edges 4` did not finish within an hour even on 64 GB.

The degree cap column isolates one cost. On the small neighbourhood graph the labelled
attempt is nearly the only source of its own star patterns, so capping degree removes
the centre and the pattern disappears. On the full graph the same patterns are supplied
by other structures below the cap, and recovery holds up better.

### The bottleneck measure

Scores a structure by the weakest link of its best-funded occurrence instead of by mean
label utility. This measure is borrowed from the weighted subgraph mining literature,
not proposed here. It is subject to the same support-dominance ceiling, with R the spread
of the edge amounts, which is far wider than any bucket table.

```bash
python scripts/e9_flow_recovery.py --dataset HI-Small --mode truth-neighborhood
python scripts/e9_flow_recovery.py --dataset HI-Small --mode full --heap 32g
```

Its edge weights are transaction amounts, so it depends directly on the US Dollar
conversion; reference values will be listed once regenerated.

### Recovery over eight typologies

```bash
python scripts/e3plus_recovery8.py --dataset HI-Small
python scripts/e3plus_recovery8.py --dataset HI-Medium \
    --global-cap 5000000 --per-start-cap 50000
```

The path miners need a search budget matched to graph size. The default budget
underreports the random and stack typologies on the larger graphs, which is a budget
artefact rather than a mining limit. These detectors are implemented in this repository
(`baselines/motif_join.py`, `baselines/motif_ext.py`); the scatter-gather detector filters
on the in/out amount ratio, so its results depend on the US Dollar conversion.

### Detection under strict induction

Fits everything on time steps up to 34 and evaluates on steps 35 to 49, split into a
stable stretch and the stretch after the market shift at step 43.

```bash
python scripts/e4plus_gnn_drift.py --seeds 10 --trees 100 --gnn gcn sage --require-gnn
```

`--require-gnn` fails immediately rather than quietly writing a table with the random
forest row alone. The driver also refuses to overwrite a table that already holds
neural rows. On Windows, `run_e4_gnn.bat` installs the CPU wheels, runs, and grades.

| Method | illicit F1 | stable 35-42 | after shift 43-49 | retention |
|---|---|---|---|---|
| random forest, 165 features | 0.826 | 0.913 | 0.032 | 0.035 |
| GraphSAGE | 0.692 | 0.769 | 0.011 | 0.015 |
| graph convolutional network | 0.531 | 0.599 | 0.000 | 0.000 |

Under a protocol with no leakage the random forest beats both neural baselines, and
every classifier collapses across the shift. The CSV records the torch build and device
in an `environment` column, because a neural F1 does not reproduce across devices; the
committed row was produced by a CPU-only build, which is deterministic under the seeds.

### Account-level detection

```bash
python scripts/e7_actor_hus_detection.py --min-util 100     # about 25 minutes
```

The pattern vocabulary is mined on the training subgraph and frozen, and every
structural feature is causal, using only snapshots at or before the step being scored.
This is a negative result and the paper reports it: mined patterns add nothing over
degree, and the degree cap that makes mining feasible is what removes the signal.
Degree lifts F1 by 0.183 on the full graph and by -0.028 after the cap, which drops 66%
of the edges, and a laundering hub is by definition high degree.

### Figures

```bash
python scripts/make_figures_paper3.py --results <dir with e9 CSVs> --e8 <e8_utility_contrast.csv>
```

Draws every value from the result files and refuses files written before the order
statistics existed. Writes vector PDFs to `--out` (the record directory under the
wrapper, else `experiments/figures/`).

## 5. Determinism

Utility buckets are fixed quantiles fitted on the training window and frozen; the miner
is deterministic; every learned baseline runs under a fixed seed on CPU. The recovery
rates, pattern counts, variance decompositions and F1 values reproduce exactly: the
utility-contrast experiment, re-run on an Apple M5 under JDK 26, matched every one of the
159 values of its committed CSVs (the environment that produced those CSVs was not
recorded). Only wall-clock
timing varies, so no timing figure should be read as a claim.

The reference timings in the paper were measured on an Intel Core i7-10750H with 64 GB
of RAM running Windows 11.
