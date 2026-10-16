"""No tool that builds tables, figures or checks of results may read the probe timing tree."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREE = "results-probe-timing"
# the recorder writes the tree, probes run under it, and tests describe it
ALLOWED = {"experiments/provenance/run_and_record.py"}
ALLOWED_DIRS = ("experiments/probes/", "experiments/tests/")


def offenders(root: Path) -> list[str]:
    found = []
    for path in sorted((root / "experiments").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in ALLOWED or rel.startswith(ALLOWED_DIRS):
            continue
        if TREE in path.read_text(encoding="utf-8", errors="replace"):
            found.append(rel)
    return found


def test_no_result_tool_names_the_probe_timing_tree():
    assert offenders(ROOT) == []


def test_the_check_finds_a_tool_that_names_the_tree(tmp_path):
    scripts = tmp_path / "experiments" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "table.py").write_text(f'ROWS = "{TREE}/x/timing.csv"\n')
    (scripts / "clean.py").write_text('ROWS = "results/x/timing.csv"\n')
    probes = tmp_path / "experiments" / "probes"
    probes.mkdir()
    (probes / "p.py").write_text(f'OUT = "{TREE}"\n')
    assert offenders(tmp_path) == ["experiments/scripts/table.py"]
