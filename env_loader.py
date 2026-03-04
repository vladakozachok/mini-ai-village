import os
from pathlib import Path

_LOADED_PATHS: set[Path] = set()


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(env_path: Path | None = None) -> Path:
    path = (env_path or (Path(__file__).resolve().parent / ".env")).resolve()
    if path in _LOADED_PATHS or not path.exists():
        return path

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = _strip_matching_quotes(value.strip())
        os.environ.setdefault(key, value)

    _LOADED_PATHS.add(path)
    return path
