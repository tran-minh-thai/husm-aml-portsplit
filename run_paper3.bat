@echo off
setlocal
REM run_paper3.bat -- run all experiments on Windows (double-click, or from cmd).
REM Args pass through to run_all_paper3.py, e.g.:  run_paper3.bat --include-medium
REM Set NO_PAUSE=1 to run inside a script without the wait-for-key at the end.

set "HERE=%~dp0"

set "PY=%USERPROFILE%\.venvs\husm-aml-portsplit\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Using Python: %PY%

REM ensure the core Python deps are present in this venv (first run installs them)
"%PY%" -c "import pandas, numpy, sklearn, scipy, networkx" 2>nul
if errorlevel 1 (
    echo Installing Python dependencies into the venv...
    "%PY%" -m pip install -r "%HERE%requirements.txt"
)
REM note: the GNN baseline in E4+ also needs torch-geometric (install separately, needs a GPU).

"%PY%" "%HERE%experiments\scripts\run_all_paper3.py" %*

if not defined NO_PAUSE pause
