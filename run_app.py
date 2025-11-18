import os
from dotenv import load_dotenv
import subprocess

# 1. Cargar variables del .env
load_dotenv()

# 2. Evitar que Streamlit pida email
os.environ["STREAMLIT_ASSUME_YES"] = "true"

# 3. Ruta del ejecutable de streamlit
STREAMLIT_PATH = "/home/leogurdu/.local/share/virtualenvs/documentation-helper-27iDdxKz-python3.12/bin/streamlit"

# 4. Archivo principal
MAIN_FILE = "main.py"

# 5. Ejecutar streamlit
subprocess.run([STREAMLIT_PATH, "run", MAIN_FILE])
