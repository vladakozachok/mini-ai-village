import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Shared filesystem paths
LOG_DIR = PROJECT_ROOT / "logs"
WEB_APP_DIR = PROJECT_ROOT / "web-app"
BACKEND_DIR = WEB_APP_DIR / "backend"
FRONTEND_DIR = WEB_APP_DIR / "frontend"

# Backend runtime defaults
API_HOST = os.getenv("MINI_AI_VILLAGE_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("MINI_AI_VILLAGE_API_PORT", "8000"))
FRONTEND_DEV_ORIGIN = os.getenv(
    "MINI_AI_VILLAGE_FRONTEND_ORIGIN",
    "http://localhost:5173",
)


def ensure_shared_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
