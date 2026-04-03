@echo off
chcp 65001 >nul 2>&1

:: ── 首次运行：检查依赖 ────────────────────────────────────────
python -c "import langgraph" >nul 2>&1
if %errorlevel% neq 0 (
    echo [setup] 依赖未安装，先执行安装...
    call "%~dp0setup.bat"
    if %errorlevel% neq 0 ( pause & exit /b 1 )
)

:: ── 启动（所有参数透传给 launch.py）──────────────────────────
python "%~dp0launch.py" %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
