import os
from dotenv import load_dotenv

load_dotenv()

SNAPSHOTS_DIR = "snapshots"
if not os.path.exists(SNAPSHOTS_DIR):
    os.makedirs(SNAPSHOTS_DIR)

PYTHON_ENGINE_URL = os.getenv("PYTHON_ENGINE_URL", "http://127.0.0.1:8000")
