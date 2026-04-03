@echo off
cd /d "%~dp0"
set FLAGS_use_mkldnn=0
set FLAGS_use_gpu=0
python omniparser.py %*
pause
