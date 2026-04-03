@echo off
chcp 65001 >nul 2>&1

set SCRIPT_DIR=%~dp0
set OUT=%SCRIPT_DIR%data\nav_graph.png

python "%SCRIPT_DIR%tools\visualize_nav.py" --out "%OUT%"
if %errorlevel% neq 0 (
    echo.
    echo [错误] 生成失败，请确认已安装 matplotlib：pip install matplotlib
    pause
    exit /b 1
)

echo.
echo 正在打开: %OUT%
start "" "%OUT%"

exit /b 0
