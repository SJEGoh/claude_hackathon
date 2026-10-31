"""Operational limits from the organisers' workbook (data/challenge2_operational_constraints.xlsx).

The workbook is the single source for these numbers; nothing here is hardcoded except how we apply them to a
minute-based arrival plan. Its airport figures are SYNTHETIC, scaled by the organisers from flights.csv.

Assumptions the workbook leaves open (stated in the README, as its rules_of_play sheet asks):
  FINAL_APPROACH_KT  converts the wake table's nautical miles into whole minutes, rounded up for safety. The
                     workbook's min_time_s column is for same-runway departures, which this project does not plan.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path

import openpyxl
from pydantic import BaseModel

WORKBOOK = Path(__file__).parent / "data" / "challenge2_operational_constraints.xlsx"
FINAL_APPROACH_KT = 140
WAKE_CATEGORIES = ("SUPER", "HEAVY", "LARGE", "SMALL")


class AirportLimits(BaseModel):
    iata: str
    size_class: str
    active_runways: int
    gates: int
    max_runway_ops_per_hour: int
    min_turnaround_min: int
    spare_gates: int


class Limits(BaseModel):
    source: str
    sha256: str
    airports: dict[str, AirportLimits]
    wake_nm: dict[tuple[str, str], int]  # (leader, follower) -> minimum separation in nautical miles
    final_reserve_min: int
    contingency_pct: float
    hold_kg_per_hour: dict[str, int]  # "narrowbody" / "widebody"
    spare_fuel_min: int
    flight_crew_max_duty_h: int
    ground_shift_h: int
    overtime_multiplier: float
    gate_change_notice_min: int

    def airport(self, iata: str) -> AirportLimits:
        return self.airports[iata]

    def wake_min(self, leader: str, follower: str) -> int:
        """Whole minutes needed between a leader and its follower on final, rounded up."""
        return math.ceil(self.wake_nm[(leader, follower)] / (FINAL_APPROACH_KT / 60))


def _first_number(text: str) -> float:
    """The first number in a cell such as '2,400 kg per hour' or '8 hours; overtime costs 1.5x after that'."""
    match = re.search(r"\d[\d,]*\.?\d*", str(text))
    if match is None:
        raise ValueError(f"no number in workbook value {text!r}")
    return float(match.group().replace(",", ""))


def _rows(sheet, header_row: int) -> list[dict]:
    rows = list(sheet.iter_rows(values_only=True))
    names = rows[header_row - 1]
    return [dict(zip(names, r)) for r in rows[header_row:] if r[0] is not None]


def _wake_table(rows: list[dict]) -> dict[tuple[str, str], int]:
    def category(label: str) -> str:
        return label.split()[0].upper()

    named = [r for r in rows if r["follower"] is not None]
    other = next(r["min_separation_nm"] for r in rows if r["follower"] is None)  # the "Any other pair" row
    table = {(a, b): int(other) for a in WAKE_CATEGORIES for b in WAKE_CATEGORIES}
    table.update({(category(r["leader"]), category(r["follower"])): int(r["min_separation_nm"]) for r in named})
    return table


@lru_cache(maxsize=1)
def load_limits() -> Limits:
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    airports = {r["IATA_CODE"]: AirportLimits(
        iata=r["IATA_CODE"], size_class=r["size_class"], active_runways=r["active_runways"], gates=r["gates"],
        max_runway_ops_per_hour=r["max_runway_ops_per_hour"], min_turnaround_min=r["min_turnaround_min"],
        spare_gates=r["spare_gates_at_any_time"]) for r in _rows(wb["airport_capacity"], 3)}
    rules = {r["rule"]: r["value"] for r in _rows(wb["fuel_and_crew"], 1)}
    shift = str(rules["Ground crew shift"])
    return Limits(
        source=WORKBOOK.name, sha256=hashlib.sha256(WORKBOOK.read_bytes()).hexdigest(), airports=airports,
        wake_nm=_wake_table(_rows(wb["wake_separation"], 3)),
        final_reserve_min=int(_first_number(rules["Final fuel reserve"])),
        contingency_pct=_first_number(rules["Contingency fuel"]),
        hold_kg_per_hour={"narrowbody": int(_first_number(rules["Holding fuel burn (narrowbody)"])),
                          "widebody": int(_first_number(rules["Holding fuel burn (widebody)"]))},
        spare_fuel_min=int(_first_number(rules["Assumed spare fuel on arrival"])),
        flight_crew_max_duty_h=int(_first_number(rules["Flight crew max duty"])),
        ground_shift_h=int(_first_number(shift)),
        overtime_multiplier=_first_number(shift.split(";", 1)[1]),
        gate_change_notice_min=int(_first_number(rules["Gate change notice"])))
