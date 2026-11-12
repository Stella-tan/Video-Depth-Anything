@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo === Video-Depth-Anything Windows setup ===

REM ---- 1. Find Python 3.10 or 3.11 (torch 2.1.1 / numpy 1.24 do not support 3.12+) ----
set "PY="
py -3.10 -c "import sys" >nul 2>&1 && set "PY=py -3.10"
if not defined PY ( py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11" )
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3,10),(3,11)] else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3.10 or 3.11 is required. Install it from https://www.python.org/downloads/release/python-31011/
  echo         and tick "Add python.exe to PATH" during install, then re-run this script.
  pause & exit /b 1
)
echo Using: %PY%

REM ---- 2. Virtual environment ----
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv || ( echo [ERROR] could not create venv & pause & exit /b 1 )
)
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip

REM ---- 3. PyTorch (CUDA 12.1 build if an NVIDIA GPU is present, else CPU) ----
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
  echo NVIDIA GPU detected - installing CUDA 12.1 build of PyTorch
  set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
  set "HAS_GPU=1"
) else (
  echo No NVIDIA GPU detected - installing CPU build of PyTorch ^(inference will be slow^)
  set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
  set "HAS_GPU=0"
)
"%VPY%" -m pip install torch==2.1.1 torchvision==0.16.1 --index-url !TORCH_INDEX! || ( echo [ERROR] torch install failed & pause & exit /b 1 )

REM ---- 4. Other requirements ----
"%VPY%" -m pip install -r requirements-windows.txt || ( echo [ERROR] requirements install failed & pause & exit /b 1 )

REM ---- 5. Optional extras (failures here are not fatal) ----
if "!HAS_GPU!"=="1" (
  echo Installing optional xformers ^(faster attention^)...
  "%VPY%" -m pip install xformers==0.0.23 --no-deps || echo [WARN] xformers not installed - the model falls back to plain PyTorch attention.
)
echo Installing optional gradio ^(for app.py web demo^) and open3d ^(for --metric point clouds^)...
"%VPY%" -m pip install gradio || echo [WARN] gradio not installed
"%VPY%" -m pip install open3d || echo [WARN] open3d not installed

REM ---- 6. Sanity check ----
"%VPY%" -c "import torch, cv2, decord; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

REM ---- 7. Weights ----
call "%~dp0download_weights.bat" vits

echo.
echo === Setup finished. Try:  run_demo.bat ===
pause
