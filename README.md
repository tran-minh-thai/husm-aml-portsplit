# husm-aml-portsplit

Amount-conserving chain mining in transaction graphs: the chain miner, the scripts that
build its inputs and run every experiment, and the record of every run.

A chain is a directed path of transactions (open chain) or a directed cycle read from a marked root
(closed chain), with account and transaction labels. An occurrence is valid when every intermediate
account forwards an amount within a relative tolerance of the amount it received,
`(1 - theta) a_in <= a_out <= (1 + theta) a_in`; the pair at the root of a closed chain is not
constrained. The support of a chain is the size of its smallest image set over valid occurrences
(minimum image based support). The miner reports every chain up to a maximum length whose support
reaches a threshold, with the exact image set of every position.

In the paper the miner is called PaCoMine. Its command line is `edu.huflit.inchusm.chain.ChainMain`, whose defaults
are the configuration used in the paper (slot filter from candidate length 4, the computation of each filter pass
chosen from operation counts); `--consistency-from 2` gives PaCoMine-Always (filter on every candidate) and
`--no-consistency` gives PaCoMine-NoFilter. `--engine enumerate` runs the enumerator it is compared with.

The repository also holds the pipeline of a high-utility subgraph mining study (port-splitting),
documented in [docs/reproduce/paper3.md](docs/reproduce/paper3.md); it shares the Java engine but no
result with the chain miner.

## Layout

```
inchusm-miner/                       Java engine (Maven, package edu.huflit.inchusm)
  chain/ChainMiner.java              level-wise chain miner: prefix-suffix join, slot filter, pinned search
  chain/Consistency.java             forward and backward slot passes (sorted, indexed or chosen per pass)
  chain/ChainEnumerator.java         baseline: enumerate every valid occurrence, merge the images
  chain/ChainMain.java               command line of the chain miner
  cycles/CycleMembership.java        image sets of closed chains at theta = inf, one account label
  mutation/                          fault injection for the chain tests
experiments/
  scripts/                           one entry point per experiment (listed below)
  src/inchusm_aml/                   dataset loaders and shared code
  provenance/                        recording wrapper, machine declaration, dataset checks and download
  tests/                             tests of the Python side
third_party/grami/                   patches and build script for the GraMi baseline (GraMi itself is not included)
docs/reproduce/                      dataset checksums and the locked Java runtime
results/                             recorded runs with timing, made on the declared measurement machine
results-invariant/                   recorded runs whose values do not depend on the machine
results-probe/, results-probe-timing/  exploratory runs on a development machine; not used for any reported number
datasets/                            input data, not tracked (see Data)
```

## Requirements

- Python 3.11 or newer with the packages of `requirements.txt` (versions pinned to the recorded runs)
- JDK 17 or newer and Maven; timed runs use the runtime locked in `docs/reproduce/jdk.lock`
  (Eclipse Adoptium 25.0.4.1)
- 64 GB of RAM: the recorded runs used a Java heap of 48 GB (20 GB for `windowed_effect.py`); the scripts
  default to 16 GB (`--heap`)

## Build and test

```bash
python -m venv ~/.venvs/husm-aml-portsplit
~/.venvs/husm-aml-portsplit/bin/python -m pip install -r requirements.txt
mvn -f inchusm-miner/pom.xml package          # builds the engine and runs its tests
python -m pytest experiments/tests
python inchusm-miner/mutation/chain_mutations.py    # every injected fault must fail the tests
python inchusm-miner/mutation/cycle_mutations.py
```

On Windows the interpreter is `%USERPROFILE%\.venvs\husm-aml-portsplit\Scripts\python.exe`.

## Data

The datasets are public and are not redistributed here.

| Data | Source | Location |
|---|---|---|
| AMLworld HI-Small, LI-Small, HI-Medium, LI-Medium | Kaggle, `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` | `datasets/amlworld/{HI,LI}-{Small,Medium}_{Trans.csv,Patterns.txt}` |
| AMLworld HI-Large, LI-Large | same; `python experiments/provenance/fetch_amlworld_large.py` downloads and checks them | `datasets/amlworld/{HI,LI}-Large_{Trans.csv,Patterns.txt}` |
| AMLGentex sweden_100K_easy, sweden_100K_difficult, us_100K_difficult | Hugging Face, `datasets/AMLGentex` | `datasets/amlgentex/<name>/tx_log.csv` |

Check the files against the recorded sizes and sha256: `python experiments/provenance/verify_datasets.py`
prints OK, MISSING, SIZE or HASH for every file of `docs/reproduce/datasets.sha256` (the files under
`datasets/amlworld/` and `datasets/amlgentex/` are the ones used here; the others belong to the port-splitting
study), and `python experiments/provenance/fetch_amlworld_large.py --check-only` checks the Large files against
`docs/reproduce/datasets_large.sha256`. Then write the miner's input for every dataset and labelling:

```bash
python experiments/scripts/export_chain_input.py --source amlworld --dataset HI-Small --labeling A
```

Labelling S has one account label and one transaction label. Labelling A uses the payment format as
transaction label on AMLworld and the bank as account label on AMLGentex. The input is written to
`datasets/chain/<dataset>_<labelling>.tg` (`v ID LABEL` per account, `e SOURCE TARGET LABEL AMOUNT` per
transaction) with a JSON file of counts, label names and sha256.

## Running the miner

```bash
java -Xmx16g -cp inchusm-miner/target/inchusm-miner-0.1.0-SNAPSHOT.jar edu.huflit.inchusm.chain.ChainMain \
     --graph datasets/chain/HI-Small_S.tg --theta 0.2 --min-support 10 --max-length 5 --out patterns.tsv
```

The output has one row per frequent chain: kind (open or closed), length, account labels, transaction
labels, support and the size of every image set. Work counters, which do not depend on the machine, go
to standard output. Options: `--theta inf` switches conservation off; `--no-closed` mines open chains
only; `--min-cycle C` is the shortest closed chain; `--mark FILE` counts listed accounts in every image
set; `--accounts-out FILE` writes the accounts at intermediate positions; `--engine enumerate` runs the
enumeration baseline; `--no-consistency`, `--consistency-from L` and `--filter-impl sorted|indexed|auto`
change only how the slot filter is applied, never the output.

## Experiments

Every run goes through the recording wrapper, which writes a record directory with the outputs and a
`run_meta.json` (command, commit, working-tree state, hashes of the sources, of the engine jar and of the
data, library and runtime versions, machine):

```bash
python experiments/provenance/run_and_record.py --into results-invariant/<tree> \
    --data <input files> -- python experiments/scripts/<script>.py <arguments>
```

Timed runs add `--timing-into results/<tree>`; the wrapper refuses them on any machine other than the
one declared in `results/MEASUREMENT_MACHINE.txt`.

A few records made on the measurement machine cite a local commit that was replaced when the results were pushed
with `git pull --rebase`, so the commit is not in this repository. `results/COMMIT_EQUIVALENCE.tsv` maps each of them
to a commit with the same source code (equal `code_sha256`), written by
`python experiments/provenance/commit_equivalence.py`, which first checks its hashing against records whose commit is
present. The script of every record tree:

| Question | Script | Record tree |
|---|---|---|
| Miner tests and fault injection | `mvn package`, `inchusm-miner/mutation/chain_mutations.py` | `chain-miner-verification` |
| Miner against GraMi on random graphs | `grami_crosscheck.py` | `chain-grami-crosscheck` |
| Exactness of the slot filter for walks | `ChainMain --check-failures` | `chain-failure-check-final` |
| Miner inputs | `export_chain_input.py` | `chain-input` |
| Roles of the candidate labels | `experiments/probes/label_roles.py` | `label-roles` |
| Lengths of the planted chains | `planted_lengths.py` | `planted-lengths` |
| Choice of theta | `theta_selection.py` | `theta-selection-final` |
| Amount rules on planted and background chains | `chain_planted_rules.py` | `chain-planted-rules` |
| Laundering accounts in conserving chains | `chain_conservation_effect.py` | `chain-conservation-effect`, `chain-maxlen6`, `chain-tau-grid`, `chain-tau-grid-a` |
| Controls (degree, amount shuffles, bootstrap) | `chain_controls.py` | `chain-controls` |
| Effect over theta | `chain_theta_grid.py` | `chain-theta-grid-final` |
| Confidence intervals | `effect_intervals.py` | `effect-intervals` |
| Closed chains | `chain_cycle_effect.py`, `cycle_membership_check.py` | `chain-cycle-effect-final`, `cycle-membership-check` |
| Time windows | `windowed_effect.py`, `windowed_effect_check.py` | `windowed-effect` |
| Running time of the miner, its variants and the enumerator | `chain_performance.py` | `chain-performance` (timing in `results/`) |
| Running time against GraMi | `grami_speed.py` | `grami-speed`, `grami-speed-sharedtimer`, `grami-speed-acfix`, `grami-env-check` |
| Differences between GraMi and the miner | `grami_cache_diagnosis.py`, `grami_added_check.py`, `grami_dump_check.py` | `grami-cache-diagnosis`, `results-probe/grami-*-check` |
| Running example | `running_example.py` | `running-example` |

Scripts are in `experiments/scripts/` unless a path is given; each one documents its arguments and
outputs in its first lines. Records are never overwritten: each run writes a new directory named by
its start time.

## Default parameters

| Parameter | Value | Where |
|---|---|---|
| Tolerance `theta` | 0.2 | `--theta` of the miner and the scripts |
| Minimum support `tau` | 10 | `--min-support`, `--tau` |
| Maximum chain length `L` | 5 | `--max-length` |
| Shortest closed chain | 3 transactions | `--min-cycle` (ChainMain default) |
| Slot filter applied from candidate length | 4 | `ChainMiner.consistencyFrom`, `--consistency-from` |
| Slot pass implementation | `auto` | `--filter-impl` |
| Step budget of the first search on a closed chain | 10,000 | `ChainMiner.DFS_BUDGET` |
| Java heap | 16 GB by default; 48 GB in the recorded runs (20 GB for `windowed_effect.py`) | `--heap` of the scripts |
| Time limit per timed run | 7,200 s (`chain_performance.py`); 1,800 s (`grami_speed.py`) | `--limit` |
| Bootstrap over laundering attempts | 1,000 resamples, seed 20261015 | `chain_controls.py` |
| Degree-matched draws and amount shuffles | 200 draws, seed 20261014 | `chain_controls.py` |
| Background chains sampled for the amount rules | 200,000, seed 20261012 | `chain_planted_rules.py`, `theta_selection.py` |
| Random graphs against GraMi | 200, seed 20261016 | `grami_crosscheck.py` |

## GraMi baseline

`third_party/grami/build_grami.py` applies the patches of `third_party/grami/` to GraMi commit `d2875ae`
and compiles the variants used by `grami_speed.py`; see [third_party/grami/README.md](third_party/grami/README.md).

## License

The code written for this repository is released under the MIT License (`LICENSE`). GraMi is not
included; the patches in `third_party/grami/` modify GraMi and are subject to its license (LGPL). The
datasets are distributed by their owners under their own terms.
