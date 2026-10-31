"""Operational settings the duty officer can change (Ops settings page). Every value is read live by the
deterministic safety checks, the agents' instruments and the weather watch.

Defaults reproduce the built-in limits. Runway-condition effects are a simplified, illustrative version of
ICAO GRF (Global Reporting Format) runway condition codes: replace the table with your operator's figures.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SETTINGS_FILE = Path(os.environ.get("OPS_SETTINGS_FILE", Path(__file__).parent / "assets" / "ops_settings.json"))

# RWYCC -> (max crosswind kt, extra spacing min). 6 dry ... 0 nil braking (runway closed). Illustrative only.
RWYCC_TABLE = {6: (None, 0), 5: (29, 0), 4: (25, 0), 3: (20, 1), 2: (15, 1), 1: (10, 2), 0: (0, 0)}
RWYCC_LABEL = {6: "6 · dry", 5: "5 · wet / good", 4: "4 · good to medium", 3: "3 · medium (slippery wet, slush)",
               2: "2 · medium to poor", 1: "1 · poor (ice)", 0: "0 · nil braking: runway closed"}
APPROACH_SPACING = {"VISUAL": 0, "ILS_CAT_I": 1, "ILS_CAT_II_III": 2}
CHECKS = ["thunderstorm", "rain", "wind", "visibility", "windshear", "turbulence_icing", "freezing", "sigmet",
          "data_health"]


class OpsSettings(BaseModel):
    # Runway and approach
    rwycc: int = Field(6, ge=0, le=6)
    approach: Literal["AUTO", "VISUAL", "ILS_CAT_I", "ILS_CAT_II_III"] = "AUTO"
    max_crosswind_kt: float = 33
    max_tailwind_kt: float = 10
    # Flight crew
    min_fuel_advisory_margin_min: int = 10
    severe_hold_limit_min: int = 10
    min_connect_time_min: int = 40
    # Ground
    ramp_staffing_pct: int = Field(100, ge=25, le=100)
    allow_remote_stands: bool = True
    max_crew_overtime_min: int = 30
    # Negotiation
    time_budget_s: int = 120
    default_autonomy: Literal["AUTONOMOUS", "SUPERVISED", "STEP_BY_STEP"] = "SUPERVISED"
    # Weather watch and notifications
    watch_airports: list[str] = ["SIN", "ORD", "ATL", "DFW"]
    watch_interval_s: int = 300
    checks: dict[str, bool] = {c: True for c in CHECKS}
    gust_alert_kt: int = 25
    heavy_rain_mm_h: float = 7.6
    sigmet_radius_nm: int = 60
    push_min_severity: Literal["INFO", "WARNING", "ALARM"] = "WARNING"
    notify_sound: bool = True
    notify_desktop: bool = False
    ntfy_topic: str = ""  # phone push via https://ntfy.sh/<topic> (install the ntfy app, subscribe to the topic)

    # ---- derived effects used by the safety checks -------------------------------------------------
    def crosswind_limit(self) -> float:
        cap = RWYCC_TABLE[self.rwycc][0]
        return self.max_crosswind_kt if cap is None else min(self.max_crosswind_kt, cap)

    def runway_closed_by_condition(self) -> bool:
        return self.rwycc == 0

    def extra_spacing(self, imc: bool) -> int:
        approach = APPROACH_SPACING.get(self.approach, 1 if imc else 0)
        return approach + RWYCC_TABLE[self.rwycc][1]

    def turnaround(self, base_min: int) -> int:
        return math.ceil(base_min * 100 / self.ramp_staffing_pct)


_CACHE: dict = {}


def get() -> OpsSettings:
    mtime = SETTINGS_FILE.stat().st_mtime if SETTINGS_FILE.exists() else 0
    if _CACHE.get("mtime") != mtime:
        data = json.loads(SETTINGS_FILE.read_text()) if mtime else {}
        _CACHE.update(mtime=mtime, s=OpsSettings(**data))
    return _CACHE["s"]


def save(s: OpsSettings) -> None:
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    SETTINGS_FILE.write_text(s.model_dump_json(indent=1))
