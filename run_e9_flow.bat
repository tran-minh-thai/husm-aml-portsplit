@echo off
setlocal
REM ============================================================================
REM run_e9_flow.bat -- typology recovery under the bottleneck measure.
REM
REM Scores a structure by the weakest link of its best-funded occurrence rather
REM than by the mean utility of its vertex labels. That form is anti-monotone by
REM construction, so the miner prunes on the measure itself and needs none of the
REM upper bounds that mean utility requires. The measure is borrowed from the
REM weighted subgraph mining literature; what this experiment shows is its effect
REM on port-split transaction graphs.
REM
REM Steps 1 and 2 run the truth-neighbourhood subgraphs of HI-Small and LI-Small.
REM Step 3 runs the full HI-Small graph and needs 64 GB of RAM. A run that exceeds
REM --timeout is recorded as timed_out=1 rather than crashing the sweep.
REM
REM The jar is always rebuilt here because the bottleneck measure lives in the
REM Java engine (BottleneckSolver), not in the Python driver.
REM
REM Needs AMLworld under datasets\amlworld\.
REM
REM Run it by double-clicking, or from cmd: run_e9_flow.bat
REM Set NO_PAUSE=1 to run it from a script.
REM ============================================================================

set "HERE=%~dp0"

set "PY=%USERPROFILE%\.venvs\husm-aml-portsplit\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Using Python: %PY%

"%PY%" -c "import pandas, numpy, scipy, networkx" 2>nul
if errorlevel 1 (
    echo Installing Python dependencies...
    "%PY%" -m pip install -r "%HERE%requirements.txt"
    if errorlevel 1 goto :failed
)

echo Building the miner jar...
call mvn -f "%HERE%inchusm-miner\pom.xml" package -q
if errorlevel 1 goto :failed

set "E9F=%HERE%experiments\scripts\e9_flow_recovery.py"

echo.
echo === HI-Small truth-neighbourhood, degree caps 0/200/50 ===
"%PY%" "%E9F%" --dataset HI-Small --mode truth-neighborhood --hub-caps 0 200 50 --timeout 600
if errorlevel 1 goto :failed

echo.
echo === LI-Small truth-neighbourhood, degree caps 0/200/50 ===
"%PY%" "%E9F%" --dataset LI-Small --mode truth-neighborhood --hub-caps 0 200 50 --timeout 600
if errorlevel 1 goto :failed

echo.
echo === HI-Small full graph, degree caps 200/50 (needs 64 GB) ===
"%PY%" "%E9F%" --dataset HI-Small --mode full --hub-caps 200 50 ^
    --mer 2 --max-edges 3 --heap 32g --timeout 3600
if errorlevel 1 echo   The full-graph step failed; see the message above.

echo.
echo Done. Results written to experiments\results\:
echo   e9_flow_recovery_HI-Small_truth-neighborhood.csv
echo   e9_flow_recovery_LI-Small_truth-neighborhood.csv
echo   e9_flow_recovery_HI-Small_full.csv
echo Columns: type, recovery_rate, timed_out, hub_cap, min_flow, n_patterns,
echo money_share, attempts_present. Read money_share and n_patterns together with
echo the recovery rate: they say how much of the score the money dimension carried
echo and how large the pattern vocabulary was.
if not defined NO_PAUSE pause
exit /b 0

:failed
echo.
echo FAILED; see the message above.
if not defined NO_PAUSE pause
exit /b 1
