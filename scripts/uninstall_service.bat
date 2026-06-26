@echo off
REM Uninstall the LarkSnap Windows service.
REM
REM Usage: scripts\uninstall_service.bat

setlocal

.venv\Scripts\python.exe -m larksnap.main uninstall
if errorlevel 1 (
    echo Uninstall failed.
    exit /b 1
)

echo Service uninstalled.
exit /b 0
