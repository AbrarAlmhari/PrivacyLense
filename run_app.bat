@echo off
rem ------------------------------------------------------------------
rem  PrivacyLens - start the app by double-clicking this file (Windows).
rem  It looks for Python in this order:
rem    1. a ".venv" folder inside the project   (normal setup, see docs\HOW_TO_RUN.md)
rem    2. the environment at %USERPROFILE%\.venvs\privacylens
rem    3. the "python" command of the computer
rem ------------------------------------------------------------------
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%USERPROFILE%\.venvs\privacylens\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo  Starting PrivacyLens...
echo  Your browser opens at http://localhost:8501 in a few seconds.
echo  Keep this window open while you use the app.
echo  To stop the app: press Ctrl+C in this window, or close it.
echo.

rem Open the browser a few seconds after the server starts.
rem (Set PRIVACYLENS_NO_BROWSER=1 to skip this, used for automated checks.)
if not defined PRIVACYLENS_NO_BROWSER start "" /b cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"

"%PY%" -m streamlit run app.py --server.port 8501

if errorlevel 1 (
  echo.
  echo  The app could not start.
  echo  Read docs\pdf\2_PrivacyLens_How_To_Run.pdf, section "If something goes wrong".
  pause
)
