# results-probe-timing

Timing of probes run on the development machine. A probe answers one question: is a change worth
making and measuring properly? Its numbers are never cited: every table and figure takes its timing
from `results/`, written only on the declared measurement machine.

Each record here is paired with a record under `results-probe/` (same run id), and its
`run_meta.json` carries `"timing_kind": "probe: ..."`. `run_and_record.py` refuses to pair this tree
with any other, and `experiments/tests/test_probe_timing_is_never_read.py` fails if a script that
builds tables or figures names this tree.
