"""Escalation channels for alarms raised by the orchestrator. None of these may block or crash a run.

Configured in .env:
  ESCALATION_WEBHOOK_URL   POST {"text": ...} (Slack / Teams / Discord incoming-webhook compatible)
  ESCALATION_DESKTOP=1     macOS desktop notification (useful for CLI and batch runs)
  DUTY_MANAGER_PHONE       shown as a tap-to-call link in the app on every alarm
  Ops settings: ntfy_topic  phone push via https://ntfy.sh/<topic> (free ntfy app, subscribe to the same topic)
The Streamlit app adds its own on-screen banner, toast and audible alert.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Literal

import requests
from pydantic import BaseModel

import config  # noqa: F401  (loads .env)


class Alarm(BaseModel):
    code: str
    severity: Literal["WARNING", "ALARM"]
    title: str
    detail: str
    flight: str
    turn: int
    raised_at: str = ""
    channels: list[str] = []


def duty_phone() -> str:
    return os.environ.get("DUTY_MANAGER_PHONE", "").strip()


def push(title: str, body: str, severity: str = "WARNING") -> list[str]:
    """Send one notification to every configured external channel. Never raises."""
    import ops_settings
    cfg, sent = ops_settings.get(), []
    url = os.environ.get("ESCALATION_WEBHOOK_URL", "").strip()
    if url:
        try:
            sent.append(f"webhook:{requests.post(url, json={'text': f'{title}. {body}'}, timeout=5).status_code}")
        except requests.RequestException as e:
            sent.append(f"webhook:failed({type(e).__name__})")
    if cfg.ntfy_topic.strip():
        try:
            r = requests.post(f"https://ntfy.sh/{cfg.ntfy_topic.strip()}", data=body.encode(), timeout=5,
                              headers={"Title": title.encode("ascii", "ignore").decode(),
                                       "Priority": {"ALARM": "urgent", "WARNING": "high"}.get(severity, "default"),
                                       "Tags": {"ALARM": "rotating_light", "WARNING": "warning"}.get(severity, "airplane")})
            sent.append(f"phone:{r.status_code}")
        except requests.RequestException as e:
            sent.append(f"phone:failed({type(e).__name__})")
    if (cfg.notify_desktop or os.environ.get("ESCALATION_DESKTOP") == "1") and sys.platform == "darwin":
        safe = f"{body}".replace('"', "'")[:240]
        subprocess.run(["osascript", "-e", f'display notification "{safe}" with title "{title[:60]}" sound name "Sosumi"'],
                       check=False, capture_output=True, timeout=5)
        sent.append("desktop")
    return sent


def dispatch(alarm: Alarm) -> Alarm:
    alarm.raised_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    alarm.channels += push(f"{alarm.severity}: {alarm.flight} {alarm.title}", f"Turn {alarm.turn}. {alarm.detail}",
                           alarm.severity)
    if duty_phone():
        alarm.channels.append(f"call:{duty_phone()}")
    alarm.channels.append("ui")
    return alarm
