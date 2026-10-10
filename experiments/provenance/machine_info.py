"""Describe the machine this runs on and, with --write, declare it the measurement machine.

    python experiments/provenance/machine_info.py            # print the description
    python experiments/provenance/machine_info.py --write    # print it and write the declaration

The declaration is results/MEASUREMENT_MACHINE.txt. Its first non-comment line is the machine's
host id, the same short hash run_and_record.py records with every run and checks before it lets
a run keep timing under results/; every other line is a comment "# key: value" describing the
machine: host name, CPU, physical cores and logical threads, RAM, operating system, architecture,
Python and its libraries, the Java runtime (version and vendor), Maven, and GPUs where they can
be read. Running --write on a machine, then committing and pushing the file, is what makes that
machine the measurement machine. The first declaration stays: writing another machine over it is
refused unless --replace is given.

Standard library only; runs on Windows, macOS and Linux. A field that cannot be read is written
as "unavailable" and the reason is listed under "# error:" lines, never guessed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "results" / "MEASUREMENT_MACHINE.txt"
LIBRARIES = ("numpy", "pandas", "scipy", "sklearn", "networkx", "matplotlib", "pytest")


def host_id() -> str:
    """Same definition as run_and_record.host_id: a short hash of the host name."""
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]


def _run(cmd: list[str]) -> str:
    """stdout and stderr of a command (java -version writes to stderr)."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}")
    return (r.stdout + r.stderr).strip()


def parse_java_properties(text: str) -> dict[str, str]:
    """java.runtime.version, java.vendor, java.vm.name from `java -XshowSettings:properties -version`."""
    out = {}
    for key in ("java.runtime.version", "java.vendor", "java.vm.name", "java.version"):
        m = re.search(rf"^\s*{re.escape(key)} = (.+)$", text, re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    return out


def parse_meminfo(text: str) -> float | None:
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 2**20, 1)
    return None


def cpu_and_ram(errors: list[str]) -> dict[str, object]:
    out: dict[str, object] = {"cpu": None, "cpu_physical_cores": None,
                              "cpu_logical_threads": os.cpu_count(), "ram_gb": None}
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            out["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                out["ram_gb"] = round(status.ullTotalPhys / 2**30, 1)
            out["cpu_physical_cores"] = int(_run(["powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"]))
        elif system == "Darwin":
            out["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            out["cpu_physical_cores"] = int(_run(["sysctl", "-n", "hw.physicalcpu"]))
            out["ram_gb"] = round(int(_run(["sysctl", "-n", "hw.memsize"])) / 2**30, 1)
        else:
            text = Path("/proc/cpuinfo").read_text()
            m = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
            out["cpu"] = m.group(1).strip() if m else None
            cores = {(a, b) for a, b in re.findall(r"physical id\s*:\s*(\d+)\s*\n(?:.*\n)*?core id\s*:\s*(\d+)", text)}
            out["cpu_physical_cores"] = len(cores) or None
            out["ram_gb"] = parse_meminfo(Path("/proc/meminfo").read_text())
    except Exception as e:  # recorded, not hidden
        errors.append(f"cpu/ram: {type(e).__name__}: {e}")
    return out


def gpus(errors: list[str]) -> str | None:
    try:
        system = platform.system()
        if system == "Windows":
            names = _run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_VideoController).Name"]).splitlines()
        elif system == "Darwin":
            names = re.findall(r"Chipset Model:\s*(.+)", _run(["system_profiler", "SPDisplaysDataType"]))
        else:
            names = [l.split(": ", 1)[-1] for l in _run(["lspci"]).splitlines() if "VGA" in l or "3D" in l]
        return "; ".join(n.strip() for n in names if n.strip()) or None
    except Exception as e:
        errors.append(f"gpu: {type(e).__name__}: {e}")
        return None


def java_and_maven(errors: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    try:
        props = parse_java_properties(_run(["java", "-XshowSettings:properties", "-version"]))
        out["java_runtime_version"] = props.get("java.runtime.version")
        out["java_vendor"] = props.get("java.vendor")
        out["java_vm"] = props.get("java.vm.name")
    except Exception as e:
        errors.append(f"java: {type(e).__name__}: {e}")
    try:
        mvn = "mvn.cmd" if platform.system() == "Windows" else "mvn"
        out["maven"] = _run([mvn, "-v"]).splitlines()[0]
    except Exception as e:
        errors.append(f"maven: {type(e).__name__}: {e}")
    return out


def library_versions() -> str:
    parts = []
    for name in LIBRARIES:
        try:
            parts.append(f"{name} {getattr(__import__(name), '__version__', 'unknown')}")
        except Exception:
            parts.append(f"{name} absent")
    return ", ".join(parts)


def describe() -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    os_desc = f"{platform.system()} {platform.release()} ({platform.version()})"
    if platform.system() == "Windows":
        os_desc += f" {platform.win32_edition() or ''}".rstrip()
    elif platform.system() == "Darwin":
        os_desc = f"macOS {platform.mac_ver()[0]} ({platform.version()})"
    desc: dict[str, object] = {
        "declared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        **cpu_and_ram(errors),
        "gpu": gpus(errors),
        "os": os_desc,
        "architecture": platform.machine(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "python_libraries": library_versions(),
        **java_and_maven(errors),
    }
    return desc, errors


def render(hid: str, desc: dict[str, object], errors: list[str]) -> str:
    lines = ["# Measurement machine of this project, written by experiments/provenance/machine_info.py.",
             "# The line below is the host id that run_and_record.py checks; the rest describes it.",
             hid]
    lines += [f"# {k}: {'unavailable' if v is None else v}" for k, v in desc.items()]
    lines += [f"# error: {e}" for e in errors]
    return "\n".join(lines) + "\n"


def declared_ids(text: str) -> list[str]:
    """Host ids declared in a declaration file: its non-empty, non-comment lines (as run_and_record reads it)."""
    return [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def write(text: str, hid: str, path: Path = DECLARATION, replace: bool = False) -> None:
    if path.exists() and not replace:
        ids = declared_ids(path.read_text(encoding="utf-8"))
        if ids and hid not in ids:
            raise SystemExit(f"refusing to write {path.name}: it declares {ids}, "
                             f"and this machine's host id is {hid}. The measurement machine is the one that "
                             f"made the first run; pass --replace only to change it on purpose.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help=f"write {DECLARATION.relative_to(ROOT).as_posix()}")
    ap.add_argument("--replace", action="store_true", help="replace a declaration of another machine")
    args = ap.parse_args()
    desc, errors = describe()
    text = render(host_id(), desc, errors)
    sys.stdout.write(text)
    if args.write:
        write(text, host_id(), replace=args.replace)
        print(f"\nwrote {DECLARATION.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
