@echo off
cd /d "%~dp0"
start /min cmd /c "call venv\Scripts\activate && streamlit run app.py"