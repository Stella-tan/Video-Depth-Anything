@echo off
REM Usage: download_weights.bat [vits] [vitb] [vitl] [metric_vits] [metric_vitb] [metric_vitl]
REM With no arguments downloads vits (~110 MB) and vitl (~1.5 GB).
setlocal EnableDelayedExpansion
cd /d "%~dp0"
if not exist checkpoints mkdir checkpoints
set "LIST=%*"
if "%LIST%"=="" set "LIST=vits vitl"
for %%M in (%LIST%) do (
  set "NAME=%%M"
  set "PREFIX=video_depth_anything"
  set "REPO=Video-Depth-Anything"
  if /i "!NAME:~0,7!"=="metric_" (
    set "NAME=!NAME:~7!"
    set "PREFIX=metric_video_depth_anything"
    set "REPO=Metric-Video-Depth-Anything"
  )
  if /i "!NAME!"=="vits" set "SIZE=Small"
  if /i "!NAME!"=="vitb" set "SIZE=Base"
  if /i "!NAME!"=="vitl" set "SIZE=Large"
  set "FILE=checkpoints\!PREFIX!_!NAME!.pth"
  if exist "!FILE!" (
    echo Already have !FILE!
  ) else (
    echo Downloading !FILE! ...
    curl.exe -L --fail -o "!FILE!.part" "https://huggingface.co/depth-anything/!REPO!-!SIZE!/resolve/main/!PREFIX!_!NAME!.pth" && move /y "!FILE!.part" "!FILE!" >nul || echo [ERROR] download of !FILE! failed
  )
)
