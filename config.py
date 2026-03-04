from pathlib import Path

from env_loader import load_env_file

load_env_file()

PROJECT_ROOT = Path(__file__).resolve().parent

LOG_DIR = PROJECT_ROOT / "logs"

def ensure_shared_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
