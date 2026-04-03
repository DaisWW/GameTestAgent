@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  Agent TARS v0.3.0 - Install Script
echo ============================================
echo.

:: --- Check Node.js ---
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org ^(>= v22^)
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('node --version') do set NODE_VER=%%v
echo [OK] Node.js: %NODE_VER%
for /f "tokens=*" %%v in ('npm --version') do set NPM_VER=%%v
echo [OK] npm:     %NPM_VER%
echo.

:: --- Check if already installed ---
where agent-tars >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%v in ('agent-tars --version 2^>^&1') do set AT_VER=%%v
    echo [OK] agent-tars already installed: !AT_VER!
    echo [INFO] Skipping install. Delete and re-run to reinstall.
    goto :done
)

:: --- Try install with official registry first ---
echo [INFO] Trying official npm registry...
echo [INFO] This may take 3-10 minutes depending on network...
echo.
npm install -g @agent-tars/cli@latest --progress true --loglevel info
if %ERRORLEVEL% EQU 0 goto :verify

:: --- Fallback: Chinese mirror (npmmirror.com) ---
echo.
echo [WARN] Official registry failed. Trying Chinese mirror (npmmirror.com)...
echo.
npm install -g @agent-tars/cli@latest --registry https://registry.npmmirror.com --progress true --loglevel info
if %ERRORLEVEL% EQU 0 goto :verify

:: --- Fallback 2: cnpm ---
echo.
echo [WARN] npmmirror failed. Trying cnpm...
where cnpm >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    cnpm install -g @agent-tars/cli@latest
    if %ERRORLEVEL% EQU 0 goto :verify
)

echo [FAIL] All install methods failed. Please check your network.
echo        Manual command: npm install -g @agent-tars/cli@latest --registry https://registry.npmmirror.com
pause
exit /b 1

:verify
echo.
where agent-tars >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Install finished but agent-tars command not found.
    echo [INFO] Try restarting the terminal and run run.bat
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('agent-tars --version 2^>^&1') do set AT_VER=%%v
echo [OK] agent-tars installed: !AT_VER!

:done
echo.
echo ============================================
echo  Install complete! Now run: run.bat
echo ============================================
echo.
pause
endlocal
