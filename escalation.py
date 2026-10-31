"""Escalation channels for alarms raised by the orchestrator. None of these may block or crash a run.

Configured in .env:
  ESCALATION_WEBHOOK_URL   POST {"text": ...} (Slack / Teams / Discord incoming-webhook compatible)
  ESCALATION_DESKTOP=1     macOS desktop notification (useful for CLI and batch runs)
  DUTY_MANAGER_PHONE       shown as a tap-to-call link in the app on every alarm
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


def dispatch(alarm: Alarm) -> Alarm:
    alarm.raised_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = f"[{alarm.severity}] {alarm.flight} turn {alarm.turn}: {alarm.title}. {alarm.detail}"
    url = os.environ.get("ESCALATION_WEBHOOK_URL", "").strip()
    if url:
        try:
            r = requests.post(url, json={"text": text}, timeout=5)
            alarm.channels.append(f"webhook:{r.status_code}")
        except requests.RequestException as e:
            alarm.channels.append(f"webhook:failed({type(e).__name__})")
    if os.environ.get("ESCALATION_DESKTOP") == "1" and sys.platform == "darwin":
        safe = text.replace('"', "'")[:240]
        subprocess.run(["osascript", "-e", f'display notification "{safe}" with title "ATM negotiation {alarm.severity}" '
                        f'sound name "Sosumi"'], check=False, capture_output=True, timeout=5)
        alarm.channels.append("desktop")
    if duty_phone():
        alarm.channels.append(f"phone:{duty_phone()}")
    alarm.channels.append("ui")
    return alarm
