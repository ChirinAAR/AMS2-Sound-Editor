@echo off
setlocal
cd /d "%~dp0"

where pythonw >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    start "" pythonw "%~dp0main.py"
    exit /b 0
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python "%~dp0main.py"
    exit /b 0
)

msg * "No se encontro Python instalado en el PATH de Windows. Instala Python 3.10+ junto con numpy para abrir AMS2 Sound Porter Studio."
exit /b 1
