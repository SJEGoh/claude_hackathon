"""Failure analysis: every negotiation that did not end in an agreed, ratified plan, explained from the logs.

Reads the hash-chained consensus logs (logs/*.jsonl) and, for each failed run, finds:
  - root cause: which hard limits the agents' plans kept breaking, and which domains collided
  - the deadlock point: the last plan on the table and exactly what it broke
  - cost of failure: the deterministic plan vs the best plan the AI produced (USD and CO2)
and supports a what-if replay with operational levers an airline actually controls (extra fuel uplift, ramp
staffing, one more free stand). Regulatory minimums such as the final fuel reserve are never relaxed.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

LOG_DIR = Path(__file__).parent / "logs"
CO2_PER_KG_FUEL = 3.16  # kg CO2 per kg of jet fuel burned (ICAO / IPCC emission factor)

# Hard-check failure text -> a readable cause. Order matters: first match wins.
CAUSES = [
    ("Fuel reserve", ("EFOB", "FRF", "fuel")),
    ("Wake separation", ("<", "min:")),
    ("Runway capacity", ("arrivals in the hour",)),
    ("Runway closed / wind", ("closed", "crosswind", "tailwind", "RWYCC")),
    ("Stand availability", ("free from", "next user", "stand", "EIBT")),
    ("Gate-change notice", ("notified", "notice")),
    ("Flight crew duty", ("duty",)),
    ("Holding in severe weather", ("SEV",)),
]
DOMAIN_LABEL = {"ATC": "ATC", "PILOT": "Flight Crew", "GROUND": "AOCC"}


def cause_of(detail: str) -> str:
    for label, keys in CAUSES:
        if any(k in detail for k in keys):
            return label
    return "Other"


class PlanCheck(BaseModel):
    turn: int
    plan_id: int
    proposal: dict
    violations: dict[str, list[str]]  # domain -> hard-check failures
    holding_fuel_kg: Optional[float] = None

    @property
    def safe(self) -> bool:
        return not any(self.violations.values())


class Run(BaseModel):
    file: str
    flight: str
    route: str
    date: str
    dot_row: Optional[int]
    weather_mode: str
    weather: str
    status: str
    plan_source: str
    reason: Optional[str]
    final: dict
    cost: Optional[float]
    reference_cost: Optional[float]
    turns: int
    alarms: list[str]
    overrides: int
    objections: list[str]
    checks: list[PlanCheck]
    started: str

    @property
    def failed(self) -> bool:
        return self.status != "CONSENSUS"


def load_runs() -> list[Run]:
    runs = []
    for f in sorted(LOG_DIR.glob("*.jsonl"), reverse=True):
        if f.name.startswith(("chat_", "EMERGENCY_")):
            continue
        try:
            rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        scen = next((r["payload"] for r in rows if r["kind"] == "scenario"), None)
        final = next((r["payload"] for r in rows if r["kind"] == "decision_final"), None)
        if not scen or not final:
            continue
        checks = []
        for r in rows:
            if r["kind"] == "validation":
                p = r["payload"]
                checks.append(PlanCheck(
                    turn=r["turn"], plan_id=p.get("plan_id", 0), proposal=p.get("proposal", {}),
                    violations={d: p[d]["violations"] for d in ("ATC", "PILOT", "GROUND") if d in p and p[d]["violations"]},
                    holding_fuel_kg=p.get("PILOT", {}).get("metrics", {}).get("holding_fuel_kg")))
        runs.append(Run(
            file=f.name, flight=scen.get("flight", "?"), route=scen.get("route", ""), date=scen.get("date", ""),
            dot_row=scen.get("dot_row"), weather_mode=scen.get("weather_mode", "HISTORICAL"),
            weather=scen.get("weather_attribution", ""), status=final.get("status", "?"),
            plan_source=final.get("plan_source", ""), reason=final.get("reason"), final=final.get("proposal", {}),
            cost=final.get("cost"), reference_cost=final.get("reference_cost"),
            turns=max([r["turn"] for r in rows] or [0]),
            alarms=[r["payload"].get("code", "") for r in rows if r["kind"] == "alarm"],
            overrides=sum(1 for r in rows if r["kind"] == "safety_override"),
            objections=[r["payload"].get("text", "") for r in rows if r["kind"] == "stakeholder_objection"],
            checks=checks, started=rows[0]["ts"]))
    return runs


class Autopsy(BaseModel):
    primary_cause: str
    causes: dict[str, int]
    collision: str
    deadlock_point: str
    best_ai_plan: Optional[dict]
    best_ai_safe: bool


def autopsy(run: Run) -> Autopsy:
    causes = Counter(cause_of(d) for c in run.checks for v in c.violations.values() for d in v)
    if "AGENT_FAILURE" in run.alarms or "AGENT_TIMEOUT" in run.alarms:
        causes["AI agents unavailable"] += 3
    if "DEADLOCK" in run.alarms:
        causes["Deadlock (no movement)"] += 1
    if "TIME_BUDGET" in run.alarms:
        causes["Time budget exceeded"] += 1
    if run.objections:
        causes["Stakeholders rejected the plan"] += len(run.objections)
    if run.overrides:
        causes["Unsafe ACCEPT overridden"] += run.overrides
    primary = causes.most_common(1)[0][0] if causes else "Unknown"
    # Collision: the plan ATC tabled vs the domains whose hard limits it broke
    broke = Counter(d for c in run.checks for d in c.violations)
    pairs = [f"ATC's plans vs {DOMAIN_LABEL[d]} ({n}×)" for d, n in broke.most_common() if d != "ATC"]
    if "ATC" in broke:
        pairs.append(f"ATC's own limits ({broke['ATC']}×)")
    last = run.checks[-1] if run.checks else None
    if last and last.violations:
        p = last.proposal
        point = (f"Turn {last.turn}, plan #{last.plan_id} (slot {p.get('landing_slot_min')}, RWY {p.get('runway')}, "
                 f"stand {p.get('gate')}) broke: " + "; ".join(f"{DOMAIN_LABEL.get(d, d)}: {v[0]}"
                                                            for d, v in last.violations.items()))
    elif last:
        point = f"Turn {last.turn}: plan #{last.plan_id} passed every hard check but was not agreed or ratified."
    else:
        point = "No plan ever reached validation (the agents did not answer)."
    safe = [c for c in run.checks if c.safe]
    best = (min(safe, key=lambda c: c.holding_fuel_kg or 0) if safe else
            min(run.checks, key=lambda c: sum(len(v) for v in c.violations.values())) if run.checks else None)
    return Autopsy(primary_cause=primary, causes=dict(causes), collision="; ".join(pairs) or "none recorded",
                   deadlock_point=point, best_ai_plan=best.proposal if best else None, best_ai_safe=bool(safe))
