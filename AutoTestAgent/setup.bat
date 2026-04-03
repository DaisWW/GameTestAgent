@echo off
chcp 65001 >nul 2>&1

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║        AutoTestAgent  Windows Setup          ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ── [1/3] 检查 Python ─────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    echo         https://www.python.org/downloads/
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [1/3] Python %PYVER% OK

:: ── [2/3] 安装依赖 ────────────────────────────────────────────
echo [2/3] 安装依赖（可能需要几分钟）...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] 依赖安装失败，请检查网络或 requirements.txt
    pause & exit /b 1
)
echo [2/3] 依赖安装完成

:: ── [3/3] 初始化配置 ───────────────────────────────────────────
if not exist ".env" (
    copy .env.example .env >nul
    echo [3/3] .env 已从模板创建，请填写 LLM_API_KEY 和 GAME_PACKAGE
) else (
    echo [3/3] .env 已存在，跳过
)
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║  安装完成！编辑 .env 后执行：                ║
echo  ║    python main.py --task "你的测试任务"       ║
echo  ╚══════════════════════════════════════════════╝
echo.
pause
