"""Point d'entrée : lance le dashboard Streamlit."""
import subprocess
import sys
from pathlib import Path

dashboard = Path(__file__).parent / "dashboard" / "app.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard)], check=True)
