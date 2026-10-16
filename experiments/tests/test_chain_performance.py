"""The performance runner: JVM lock, child process outcomes, log redaction and an end-to-end run."""
import random
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chain_performance as cp  # noqa: E402

JAVA = shutil.which("java")


def test_lock_is_written_once_then_enforced(tmp_path):
    lock = tmp_path / "jdk.lock"
    me = {"java.runtime.version": "25.0.4.1+1", "java.vendor": "Eclipse Adoptium"}
    assert cp.check_lock(me, lock) is None and lock.is_file()
    assert cp.check_lock(me, lock) is None
    assert "differs" in cp.check_lock(dict(me, **{"java.vendor": "Microsoft"}), lock)
    assert "differs" in cp.check_lock(dict(me, **{"java.runtime.version": "25.0.4+7"}), lock)
    assert "could not read" in cp.check_lock({"java.vendor": "x"}, lock)


def test_run_reports_ok_failure_and_timeout(tmp_path):
    # the interpreter itself: on Windows a virtual environment's python.exe is a launcher that starts it as a child
    py = getattr(sys, "_base_executable", sys.executable)
    ok = cp.run([py, "-c", "x = bytearray(50_000_000)"], tmp_path / "a.txt", 30)
    assert ok["status"] == "ok" and ok["peak_bytes"] and ok["peak_bytes"] > 50_000_000
    assert cp.run([py, "-c", "raise SystemExit(3)"], tmp_path / "b.txt", 30)["status"] == "failed (exit 3)"
    slow = cp.run([py, "-c", "import time; time.sleep(30)"], tmp_path / "c.txt", 1)
    assert slow["status"] == "timeout" and slow["wall_s"] < 10 and slow["peak_bytes"] is None


def test_kept_log_has_no_timing_and_the_timing_record_has_it(tmp_path, monkeypatch):
    out, timing = tmp_path / "rec", tmp_path / "tim"
    out.mkdir()
    timing.mkdir()
    monkeypatch.setattr(cp, "OUT", out)
    monkeypatch.setattr(cp, "TIMING", timing)
    raw = out / "raw.txt"
    raw.write_text("candidates=1\nmining elapsed ms: 1234 (machine-dependent)\n")
    cp.keep_log(raw, "miner")
    assert "1234" not in (out / "log_miner.txt").read_text() and "1234" in (timing / "log_miner.txt").read_text()
    assert not raw.exists()


@pytest.mark.skipif(JAVA is None, reason="needs java")
def test_the_jvm_binary_comes_from_java_home():
    identity = cp.jvm_identity()
    exe = Path(cp.jvm_executable(identity))
    assert exe.is_file() and exe.parent.parent == Path(identity["java.home"])


def tiny_graph(path: Path, seed: int = 7) -> None:
    rnd = random.Random(seed)
    lines = ["# directed transaction graph: v ID LABEL / e SOURCE TARGET LABEL AMOUNT"]
    lines += [f"v {v} 0" for v in range(40)]
    for _ in range(400):
        s = rnd.randrange(40)
        t = (s + 1 + rnd.randrange(39)) % 40
        lines.append(f"e {s} {t} {rnd.randrange(2)} {rnd.choice([80, 90, 100, 110, 120])}")
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.skipif(JAVA is None or not cp.JAR.is_file(), reason="needs java and the built miner jar")
def test_three_engines_agree_end_to_end_and_a_different_table_is_refused(tmp_path, monkeypatch):
    chain, out, timing = tmp_path / "chain", tmp_path / "rec", tmp_path / "tim"
    for d in (chain, out, timing):
        d.mkdir()
    tiny_graph(chain / "T_A.tg")
    monkeypatch.setattr(cp, "CHAIN", chain)
    monkeypatch.setattr(cp, "OUT", out)
    monkeypatch.setattr(cp, "TIMING", timing)
    monkeypatch.setattr(cp, "LOCK", tmp_path / "jdk.lock")
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "T", "--labeling", "A", "--tau", "2", "--max-length", "4",
                                      "--heap", "1g", "--limit", "120"])
    real_lock = Path(cp.__file__).resolve().parents[2] / "docs" / "reproduce" / "jdk.lock"
    existed = real_lock.is_file()
    cp.main()
    assert (tmp_path / "jdk.lock").is_file() and real_lock.is_file() == existed   # the lock of the test only
    rows = (out / "performance.csv").read_text().splitlines()
    assert len(rows) == 4 and all(",ok," in r for r in rows[1:])
    assert len({r.split(",")[9] for r in rows[1:]}) == 1                     # one table digest for all engines
    assert "mining_ms" in (timing / "timing.csv").read_text()
    assert "elapsed" not in "".join(p.read_text() for p in out.glob("log_*.txt"))

    # fault: the enumeration engine run with another theta writes another table and must be refused
    monkeypatch.setitem(cp.ENGINES, "enumerate", ["--engine", "enumerate", "--theta", "0.05"])
    with pytest.raises(SystemExit, match="OUTPUT DIFFERS"):
        cp.main()


@pytest.mark.skipif(JAVA is None or not cp.JAR.is_file(), reason="needs java and the built miner jar")
def test_per_length_filter_engines_write_the_same_table(tmp_path, monkeypatch):
    chain, out = tmp_path / "chain", tmp_path / "rec"
    for d in (chain, out):
        d.mkdir()
    tiny_graph(chain / "T_A.tg")
    monkeypatch.setattr(cp, "CHAIN", chain)
    monkeypatch.setattr(cp, "OUT", out)
    monkeypatch.setattr(cp, "TIMING", None)
    monkeypatch.setattr(cp, "LOCK", tmp_path / "jdk.lock")
    engines = ["miner", "miner-from-3", "miner-from-4", "miner-from-5", "miner-no-filter"]
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "T", "--labeling", "A", "--tau", "2", "--max-length", "5",
                                      "--heap", "1g", "--limit", "120", "--engines", *engines])
    cp.main()
    rows = (out / "performance.csv").read_text().splitlines()[1:]
    assert [r.split(",")[7] for r in rows] == engines and all(",ok," in r for r in rows)
    assert len({r.split(",")[9] for r in rows}) == 1
    # the default set is unchanged by the new engines
    assert cp.DEFAULT_ENGINES == ["miner", "miner-no-filter", "enumerate"]
