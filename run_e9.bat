@echo off
setlocal
REM ============================================================================
REM run_e9.bat -- port-split typology recovery on AMLworld.
REM
REM Splits every account into an in-port and an out-port so that money-flow
REM direction lives in the vertex labels, mines the transformed graph, and checks
REM whether each mined pattern occurs inside the port-split subgraph of a labelled
REM laundering attempt.
REM
REM Steps 1 and 2 run the truth-neighbourhood subgraph of HI-Small and LI-Small
REM (about 16k vertices each, two minutes). Step 3 runs the full HI-Small graph
REM (about 915k vertices) and needs 64 GB of RAM and one to two hours.
REM
REM --utility-scale raw uses transaction amounts directly. The log setting
REM compresses the utility table until support alone decides most pattern pairs,
REM which turns the run into frequent mining in disguise; the driver prints the
REM table spread and the utility share on every run and warns below 10 percent.
REM
REM --min-util is absolute while pattern utility grows with graph size, so it is
REM scaled with the graph: 12000 on the full graph corresponds to 200 on the
REM subgraph, which is 57 times smaller. A run that exceeds --timeout is recorded
REM as timed_out=1 rather than crashing the sweep.
REM
REM Results are deterministic: the utility buckets are fixed quantiles and the
REM miner is deterministic, so there is no seed to vary and no deviation to report.
REM
REM Needs AMLworld under datasets\amlworld\ (HI-Small_Trans.csv, HI-Small_Patterns.txt
REM and the LI-Small pair). The jar is built automatically if it is missing.
REM
REM Run it by double-clicking, or from cmd: run_e9.bat
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

set "JAR=%HERE%inchusm-miner\target\inchusm-miner-0.1.0-SNAPSHOT.jar"
if not exist "%JAR%" (
    echo Building the miner jar...
    call mvn -f "%HERE%inchusm-miner\pom.xml" package -q
    if errorlevel 1 goto :failed
)

set "E9=%HERE%experiments\scripts\e9_portsplit_recovery.py"

echo.
echo === HI-Small truth-neighbourhood, raw utilities, degree caps 0/200/50 ===
"%PY%" "%E9%" --dataset HI-Small --mode truth-neighborhood ^
    --hub-caps 0 200 50 --mer 2 --max-edges 3 --min-util 200 --utility-scale raw
if errorlevel 1 goto :failed

echo.
echo === LI-Small truth-neighbourhood, raw utilities, degree caps 0/200/50 ===
"%PY%" "%E9%" --dataset LI-Small --mode truth-neighborhood ^
    --hub-caps 0 200 50 --mer 2 --max-edges 3 --min-util 200 --utility-scale raw
if errorlevel 1 goto :failed

echo.
echo === HI-Small full graph, raw utilities, degree caps 200/50 (needs 64 GB) ===
"%PY%" "%E9%" --dataset HI-Small --mode full ^
    --hub-caps 200 50 --mer 2 --max-edges 3 --min-util 12000 --utility-scale raw ^
    --heap 32g --timeout 3600
if errorlevel 1 echo   The full-graph step failed; see the message above.

echo.
echo Done. Results written to experiments\results\:
echo   e9_portsplit_recovery_HI-Small_truth-neighborhood_raw.csv
echo   e9_portsplit_recovery_LI-Small_truth-neighborhood_raw.csv
echo   e9_portsplit_recovery_HI-Small_full_raw.csv
echo Each row carries recovery_rate together with utility_scale, R, utility_share
echo and locked_pairs. A utility_share below 0.10 means the run is support-dominated.
if not defined NO_PAUSE pause
exit /b 0

:failed
echo.
echo FAILED; see the message above.
if not defined NO_PAUSE pause
exit /b 1
