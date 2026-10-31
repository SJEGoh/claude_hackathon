"""Customisable agent profiles (display name + avatar), persisted in assets/agent_profiles.json.

Defaults: pixel-art crew cropped from assets/roster.mp4 for ATC, Flight Crew and AOCC; an emoji for the
Chief of Staff. An avatar is either an emoji or a path to an image under assets/avatars/.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
AVATAR_DIR = ASSETS / "avatars"
PROFILE_FILE = ASSETS / "agent_profiles.json"

DEFAULTS = {
    "ATC": {"name": "ATC · Arrival Manager", "avatar": "assets/avatars/default_atc.png"},
    "PILOT": {"name": "Flight Crew · Captain", "avatar": "assets/avatars/default_pilot.png"},
    "GROUND": {"name": "AOCC · Turnaround", "avatar": "assets/avatars/default_ground.png"},
    "MODERATOR": {"name": "Chief of Staff", "avatar": "🧭"},
    "SYSTEM": {"name": "Safety engine", "avatar": "🛡️"},
    "user": {"name": "Duty officer", "avatar": "👤"},
}


_CACHE: dict = {}


def load() -> dict:
    mtime = PROFILE_FILE.stat().st_mtime if PROFILE_FILE.exists() else 0
    if _CACHE.get("mtime") == mtime:
        return _CACHE["data"]
    data = {k: dict(v) for k, v in DEFAULTS.items()}
    if PROFILE_FILE.exists():
        for k, v in json.loads(PROFILE_FILE.read_text()).items():
            data.setdefault(k, {}).update(v)
    for v in data.values():  # a missing image file falls back to a neutral emoji
        a = v.get("avatar", "")
        if a.startswith("assets/") and not (ROOT / a).exists():
            v["avatar"] = "🤖"
    _CACHE.update(mtime=mtime, data=data)
    return data


def save(agent: str, name: str | None = None, avatar: str | None = None, image_bytes: bytes | None = None,
         ext: str = "png") -> None:
    current = json.loads(PROFILE_FILE.read_text()) if PROFILE_FILE.exists() else {}
    entry = current.setdefault(agent, {})
    if name:
        entry["name"] = name
    if image_bytes:
        AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        path = AVATAR_DIR / f"custom_{agent.lower()}.{ext}"
        path.write_bytes(image_bytes)
        entry["avatar"] = f"assets/avatars/{path.name}"
    elif avatar:
        entry["avatar"] = avatar
    PROFILE_FILE.write_text(json.dumps(current, indent=1))


def reset(agent: str | None = None) -> None:
    if not PROFILE_FILE.exists():
        return
    current = json.loads(PROFILE_FILE.read_text())
    if agent is None:
        current = {}
    else:
        current.pop(agent, None)
    PROFILE_FILE.write_text(json.dumps(current, indent=1))


def avatar_for_streamlit(agent: str, profiles: dict | None = None):
    a = (profiles or load()).get(agent, {}).get("avatar", "🤖")
    return str(ROOT / a) if a.startswith("assets/") else a
