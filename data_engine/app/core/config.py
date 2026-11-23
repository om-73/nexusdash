import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
data_engine_env = BASE_DIR / ".env"
root_env = BASE_DIR.parent / ".env"

if data_engine_env.exists():
    load_dotenv(dotenv_path=data_engine_env)
elif root_env.exists():
    load_dotenv(dotenv_path=root_env)

SNAPSHOTS_DIR = os.path.join(str(BASE_DIR), "snapshots")
if not os.path.exists(SNAPSHOTS_DIR):
    os.makedirs(SNAPSHOTS_DIR)

PYTHON_ENGINE_URL = os.getenv("PYTHON_ENGINE_URL", "http://127.0.0.1:8000")
