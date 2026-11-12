@echo off
REM Usage: run_demo.bat [video_path] [encoder: vits|vitb|vitl]
cd /d "%~dp0"
set "VIDEO=%~1"
if "%VIDEO%"=="" set "VIDEO=assets\example_videos\davis_rollercoaster.mp4"
set "ENC=%~2"
if "%ENC%"=="" set "ENC=vits"
if not exist "checkpoints\video_depth_anything_%ENC%.pth" call download_weights.bat %ENC%
set "EXTRA="
.venv\Scripts\python.exe -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || set "EXTRA=--fp32"
.venv\Scripts\python.exe run.py --input_video "%VIDEO%" --output_dir outputs --encoder %ENC% %EXTRA%
echo Results are in the outputs folder.
pause
