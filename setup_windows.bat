@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete. In VS Code: Ctrl+Shift+P, "Python: Select Interpreter", choose venv.
pause
