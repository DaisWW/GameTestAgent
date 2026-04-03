@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================
echo  Agent TARS v0.3.0 - Launch
echo ============================================
echo.

where agent-tars >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] agent-tars not installed. Run install.bat first.
    pause
    exit /b 1
)

:: --- Model Config ---
:: Edit these values or set them as environment variables before running

set PROVIDER=volcengine
set MODEL=ep-20260402192051-l8fkg
set BASE_URL=https://ark.cn-beijing.volces.com/api/v3
set API_KEY=%AGENT_TARS_API_KEY%

if "%API_KEY%"=="" (
    echo [INFO] No API key found in env AGENT_TARS_API_KEY.
    set /p API_KEY="Enter your VolcEngine Ark API Key: "
)

if "%API_KEY%"=="" (
    echo [ERROR] API Key is required.
    pause
    exit /b 1
)

echo.
echo [INFO] Provider : %PROVIDER%
echo [INFO] Model    : %MODEL%
echo [INFO] Launching on http://localhost:8888 ...
echo [INFO] Press Ctrl+C to stop.
echo.

agent-tars --provider %PROVIDER% --model %MODEL% --apiKey %API_KEY% --port 8888 --thinking

pause
endlocal
