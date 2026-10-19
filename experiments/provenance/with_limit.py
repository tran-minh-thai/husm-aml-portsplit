"""Run a command under a wall-clock limit, killing its whole process tree when the limit is reached.

    python experiments/provenance/with_limit.py SECONDS -- command [args...]

Exit status: the command's own, or 124 when the limit was reached (as GNU timeout). The command runs in its own
process group (session on POSIX, new process group on Windows), so a Java child started by a Python driver is
stopped with it rather than left running. Meant to sit between run_and_record.py and the command, so a run that
hits the limit still leaves a record, with exit code 124 and the line printed below.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.exit("usage: with_limit.py SECONDS -- command [args...]")
    limit = float(sys.argv[1])
    cmd = sys.argv[3:]
    if os.name == "nt":
        proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        return proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        print(f"time limit of {limit:g} s reached; the command and its children were stopped", flush=True)
        return 124


if __name__ == "__main__":
    sys.exit(main())
