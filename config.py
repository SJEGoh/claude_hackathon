"""Load .env into os.environ (without overriding variables already set in the shell)."""
import os
from pathlib import Path

ENV_FILE = Path(__file__).parent / ".env"


def load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        current = os.environ.get(key, "")
        # The shell wins, unless it holds an obvious placeholder such as "sk-ant-..."
        if not current or (key.endswith("_API_KEY") and (len(current) < 40 or "..." in current)):
            os.environ[key] = value


load_env()
