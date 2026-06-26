@echo off
REM Install the LarkSnap Windows service via pywin32.
REM
REM Usage: scripts\install_service.bat

setlocal

if not exist .venv (
    echo Creating uv-managed virtual environment...
    uv venv
    if errorlevel 1 goto :error
)

echo Installing dependencies into .venv ...
uv pip install -e ".[all,service-windows]"
if errorlevel 1 goto :error

echo Registering service with the SCM ...
.venv\Scripts\python.exe -m larksnap.main install
if errorlevel 1 goto :error

echo.
echo Service installed. Start it with:
echo     sc start LarkSnap
echo or
echo     net start LarkSnap
echo.
exit /b 0

:error
echo.
echo Install failed. See messages above.
exit /b 1
