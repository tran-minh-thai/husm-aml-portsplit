# Reproducing the results

Every number in the paper comes from a script in `experiments/scripts/` and is written
to a CSV under `experiments/results/`. Those CSVs are committed, so the tables can be
checked without running anything. This guide covers the environment, the data, the
command behind each result, and the values to expect.

## 1. Environment

```bash
python -m venv ~/.venvs/husm-aml-portsplit        # Windows: %USERPROFILE%\.venvs\husm-aml-portsplit
source ~/.venvs/husm-aml-portsplit/bin/activate   # Windows:...\Scripts\activate
pip install -r requirements.txt
mvn -f inchusm-miner/pom.xml package            # JDK 17; builds the engine and runs its tests
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

This runs each step in order, times it, continues past a failure, prints a summary, and
finally grades the deterministic values with `check_results.py`. A step can exit
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
quote seed 42 alone: it gives the lowest variance share of the ten and overstates the
effect.

| Graph and table | R | forced pairs | variance from utility | Jaccard vs frequent |
|---|---|---|---|---|
| Citeseer, `N(3, 1)`, 6 labels | 2.26 | 51.1% | 8.0% | 0.825 |
| Elliptic++, `N(3, 1)`, 4 labels | 2.19 | 69.6% | 1.0% | - |
| Elliptic++, monetary, 4 buckets | 4456 | 0.0% | 33.4% | - |

"Forced pairs" is the share of pattern pairs whose order no utility table can change,
because the mean utility is trapped between the smallest and largest table entries. It
is computed for each graph from its own support distribution, and it is a lower bound.

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

`--utility-scale raw` uses transaction amounts directly. The `log` setting compresses
them until the table spread falls to about 2.7, at which point the utility dimension
carries only 2% of the score variance and the run is frequent mining in disguise. The
driver prints the spread and the variance share on every run and warns below 10%.

Recovery on HI-Small, truth-neighbourhood, raw utilities:

| Typology | no cap | cap 200 | cap 50 |
|---|---|---|---|
| gather-scatter | 1.00 | 1.00 | 0.33 |
| scatter-gather | 1.00 | 1.00 | 0.73 |
| fan-in | 0.90 | 0.88 | 0.55 |
| fan-out | 0.81 | 0.81 | 0.38 |
| stack | 0.81 | 0.79 | 0.16 |
| cycle | 0.76 | 0.76 | 0.13 |
| random | 0.54 | 0.54 | 0.05 |
| bipartite | 0.00 | 0.00 | 0.00 |

Bipartite recovery is zero at every setting because a bipartite core needs more than
three edges to be told apart from a fan, and `--max-edges 4` does not finish within an
hour even on 64 GB. That is a limit of the pattern space, not of the machine.

The degree cap column isolates one cost. On the small neighbourhood graph the labelled
attempt is nearly the only source of its own star patterns, so capping degree removes
the centre and the pattern disappears. On the full graph the same patterns are supplied
by other structures below the cap, and recovery holds up better.

### The bottleneck measure

Scores a structure by the weakest link of its best-funded occurrence instead of by mean
label utility. This measure is borrowed from the weighted subgraph mining literature,
not proposed here; it is included because its form escapes the ceiling.

```bash
python scripts/e9_flow_recovery.py --dataset HI-Small --mode truth-neighborhood
python scripts/e9_flow_recovery.py --dataset HI-Small --mode full --heap 32g
```

On the full graph it recovers gather-scatter and scatter-gather at 1.00, fan-in at 0.90
and fan-out at 0.83, and carries 47.6% of the score variance in the money dimension
against 12.9% for mean utility on raw amounts.

### Recovery over eight typologies

```bash
python scripts/e3plus_recovery8.py --dataset HI-Small
python scripts/e3plus_recovery8.py --dataset HI-Medium \
    --global-cap 5000000 --per-start-cap 50000
```

The path miners need a search budget matched to graph size. The default budget
underreports the random and stack typologies on the larger graphs, which is a budget
artefact rather than a mining limit.

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
python scripts/make_figures_paper3.py
```

Writes vector PDFs next to the manuscript, outside this repository.

## 5. Determinism

Utility buckets are fixed quantiles fitted on the training window and frozen; the miner
is deterministic; every learned baseline runs under a fixed seed on CPU. The recovery
rates, pattern counts, variance shares and F1 values reproduce exactly. Only wall-clock
timing varies, so no timing figure should be read as a claim.

The reference timings in the paper were measured on an Intel Core i7-10750H with 64 GB
of RAM running Windows 11.
