@echo off
setlocal
REM ============================================================================
REM run_e4_gnn.bat -- detection under strict induction, including the neural rows.
REM
REM Fits every model on Elliptic time steps up to 34 and evaluates on steps 35 to
REM 49, reported separately for the stable stretch and for the stretch after the
REM market shift at step 43. The random forest needs nothing beyond the base
REM requirements; the graph neural baselines need torch and torch_geometric, and
REM this script installs the CPU wheels if torch is absent.
REM
REM Needs Elliptic under datasets\elliptic1\ (features, classes and edgelist) and
REM about 8 GB of RAM.
REM
REM DEVICE. The script uses whichever torch is already installed and does not
REM change it; the device in use is printed before the run and recorded in the
REM environment column of the result CSV. Only when torch is absent does it
REM install one, and then the CPU build: it is about 200 MB instead of 2.5 GB,
REM needs no driver match, and Elliptic is small enough to finish in minutes.
REM To force CPU on a machine that has CUDA, set CUDA_VISIBLE_DEVICES= first.
REM
REM A CPU run is deterministic under the fixed seeds and reproduces the committed
REM row exactly. A CUDA run need not: torch.manual_seed does not make CUDA kernels
REM deterministic, so a GPU re-run may land slightly apart. The committed CSV was
REM produced on CPU, which is why the environment column exists.
REM
REM Run it by double-clicking, or from cmd: run_e4_gnn.bat
REM Set NO_PAUSE=1 to run it from a script.
REM ============================================================================

set "HERE=%~dp0"

set "PY=%USERPROFILE%\.venvs\husm-aml-portsplit\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Using Python: %PY%

REM --- core deps (pandas/sklearn/...) -----------------------------------------
"%PY%" -c "import pandas, numpy, sklearn, scipy" 2>nul
if errorlevel 1 (
    echo Installing core dependencies...
    "%PY%" -m pip install -r "%HERE%requirements.txt"
    if errorlevel 1 goto :failed
)

REM --- torch + torch_geometric ------------------------------------------------
REM An existing torch is left ALONE, CUDA build or not: it is what produced the
REM numbers we are backing. Only a missing one gets installed, and then CPU.
"%PY%" -c "import torch" 2>nul
if errorlevel 1 (
    echo torch not found -- installing the CPU build...
    "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto :failed
) else (
    echo torch already present -- leaving it as is.
)
"%PY%" -c "import torch_geometric" 2>nul
if errorlevel 1 (
    echo Installing PyTorch Geometric...
    "%PY%" -m pip install torch_geometric
    if errorlevel 1 goto :failed
)

REM Confirm before spending the run: the whole point is the GNN rows, so stop
REM here rather than produce another RF-only table.
"%PY%" -c "import torch, torch_geometric; print('torch', torch.__version__, '| torch_geometric', torch_geometric.__version__, '| device', 'cuda' if torch.cuda.is_available else 'cpu')"
if errorlevel 1 (
    echo.
    echo ERROR: torch_geometric still does not import, so the neural rows
    echo cannot be produced. Nothing was overwritten.
    goto :failed
)

REM --- the run ----------------------------------------------------------------
REM --require-gnn makes the driver fail rather than quietly writing RF alone.
echo.
echo Running E4+ (strict inductive: fit on steps ^<= 34; RF + GCN + GraphSAGE)...
"%PY%" "%HERE%experiments\scripts\e4plus_gnn_drift.py" --seeds 10 --trees 100 --gnn gcn sage --require-gnn
if errorlevel 1 goto :failed

REM --- grade it ---------------------------------------------------------------
REM A step can exit 0 and still hold the wrong numbers, so grade the values.
REM Expect a pass for the detection table here; the other checks only pass
REM after a full run via run_paper3.bat.
echo.
echo Grading the deterministic values...
"%PY%" "%HERE%experiments\scripts\check_results.py"

echo.
echo Done. The detection table is experiments\results\e4plus_gnn_drift.csv and
echo must now hold three rows: random forest, graph convolutional network
echo and GraphSAGE, with the torch build and device in the environment column.
if not defined NO_PAUSE pause
exit /b 0

:failed
echo.
echo FAILED -- see the message above. results\e4plus_gnn_drift.csv was NOT changed.
if not defined NO_PAUSE pause
exit /b 1
