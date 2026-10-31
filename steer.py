"""Steering a live negotiation: the duty officer's new information -> Chief of Staff synthesis -> safety firewall ->
updated scenario -> restart.

The Chief turns free text into STRUCTURED events (runway closed, stand unavailable, ramp closed, aircraft later,
fuel lower, emergency) plus guidance per specialist. The firewall is plain code: events may only add restrictions,
and any request to relax a hard limit is refused (the Chief then explains why and offers a safe alternative).
Accepted events are applied to the scenario itself, so the deterministic checks enforce them; the agents also see
the directive on their bus.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from agents.base import SAFETY_CHARTER
from scenario import Scenario

HARD_LIMITS = ["fuel_reserve", "wake_separation", "runway_capacity", "runway_closure", "wind_limits", "stand_size",
               "crew_duty", "gate_change_notice", "severe_weather_holding"]
LIMIT_LABEL = {"fuel_reserve": "final reserve + alternate fuel", "wake_separation": "wake-turbulence separation",
               "runway_capacity": "runway capacity", "runway_closure": "a weather/incident runway closure",
               "wind_limits": "crosswind/tailwind limits", "stand_size": "stand size",
               "crew_duty": "flight-crew duty limits", "gate_change_notice": "gate-change notice",
               "severe_weather_holding": "holding in severe turbulence/icing"}


class SteerEvent(BaseModel):
    kind: Literal["runway_closed", "stand_unavailable", "ramp_closed", "aircraft_later", "fuel_lower"]
    runway: str  # runway id for runway_closed, else ""
    gate: str  # stand id for stand_unavailable, else ""
    minutes: int  # how long / how much later / how many minutes of holding fuel lost


class Guidance(BaseModel):
    agent: Literal["ATC", "PILOT", "GROUND"]
    instruction: str


class SteerDirective(BaseModel):
    summary: str  # one sentence: what changed
    events: list[SteerEvent]
    guidance: list[Guidance]
    relaxations_requested: list[Literal["fuel_reserve", "wake_separation", "runway_capacity", "runway_closure",
                                        "wind_limits", "stand_size", "crew_duty", "gate_change_notice",
                                        "severe_weather_holding"]]
    emergency: Literal["NONE", "FUEL", "MEDICAL", "TECHNICAL"]


class Refusal(BaseModel):
    header: str  # 3-5 words
    explanation: str  # the exact conflict, numbers if any
    safe_alternative: str  # a revised instruction the officer can send as-is


SYNTH_TASK = """The duty officer has interrupted the negotiation with new information. Translate it into structure:
- events: ONLY new facts that restrict operations (runway closed for N minutes, a stand unavailable, ramp closed,
  the aircraft arriving N minutes later, N minutes of holding fuel lost). Use the real runway and stand ids listed.
- guidance: one instruction per affected specialist.
- relaxations_requested: list every hard limit the officer's words ask to relax, skip or override (for example
  "ignore wake separation", "use the reserve fuel", "reopen the runway despite the storm"). Do not soften them.
- emergency: set it if the officer reports a MAYDAY, medical emergency or technical failure."""


def synthesize(chief, s: Scenario, text: str, record: str) -> SteerDirective:
    content = (f"## Active flight\n{chief.public_brief(s)}\nRunways: {', '.join(r.id for r in s.runways)}. "
               f"Stands: {', '.join(g.id for g in s.gates)}.\n\n## Negotiation so far\n{record or '(none)'}\n\n"
               f"## The duty officer says\n{text}\n\n## Task\n{SYNTH_TASK}")
    reply, _ = chief._call(content, SteerDirective, 0, system=f"{chief.role}\n\n{SAFETY_CHARTER}")
    return reply


def explain_refusal(chief, text: str, refused: list[str]) -> Refusal:
    content = (f"The duty officer asked: \"{text}\"\nThe deterministic safety firewall refused it because it would "
               f"relax: {', '.join(LIMIT_LABEL[r] for r in refused)}. These are hard limits that nobody can waive.\n"
               "Explain the conflict briefly and offer one safe alternative instruction that achieves the officer's "
               "goal within the limits (for example a priority request, an emergency declaration, or a diversion).")
    reply, _ = chief._call(content, Refusal, 0, system=f"{chief.role}\n\n{SAFETY_CHARTER}")
    return reply


def firewall(s: Scenario, d: SteerDirective) -> tuple[list[SteerEvent], list[str], list[str]]:
    """-> (accepted events, refused hard-limit relaxations, notes on ignored events). Plain code, no AI."""
    accepted, notes = [], []
    for e in d.events:
        if not 1 <= e.minutes <= 240:
            notes.append(f"ignored {e.kind}: {e.minutes} min is outside 1-240")
        elif e.kind == "runway_closed" and s.runway(e.runway) is None:
            notes.append(f"ignored runway_closed: unknown runway {e.runway!r}")
        elif e.kind == "stand_unavailable" and s.gate(e.gate) is None:
            notes.append(f"ignored stand_unavailable: unknown stand {e.gate!r}")
        else:
            accepted.append(e)
    return accepted, list(dict.fromkeys(d.relaxations_requested)), notes


def apply(s: Scenario, events: list[SteerEvent], why: str) -> Scenario:
    """Events only ever add restrictions, so the hard checks get stricter, never looser."""
    s = s.model_copy(deep=True)
    now = s.decision_time_min
    for e in events:
        if e.kind == "runway_closed":
            r = s.runway(e.runway)
            r.closed_until, r.closure_reason = max(r.closed_until, now + e.minutes), f"duty officer: {why}"
        elif e.kind == "stand_unavailable":
            g = s.gate(e.gate)
            g.free_from = max(g.free_from, now + e.minutes)
        elif e.kind == "ramp_closed":
            s.ramp_closed_until = max(s.ramp_closed_until, now + e.minutes)
        elif e.kind == "aircraft_later":
            s.eta_min += e.minutes
        elif e.kind == "fuel_lower":
            s.fob_at_eta_kg -= e.minutes * s.hold_burn_kg_per_min
    return s


def describe(s: Scenario, e: SteerEvent) -> str:
    return {"runway_closed": f"RWY {e.runway} closed until {s.clock(s.decision_time_min + e.minutes)}",
            "stand_unavailable": f"stand {e.gate} unavailable until {s.clock(s.decision_time_min + e.minutes)}",
            "ramp_closed": f"ramp closed until {s.clock(s.decision_time_min + e.minutes)}",
            "aircraft_later": f"aircraft {e.minutes} min later",
            "fuel_lower": f"{e.minutes} min of holding fuel lost"}[e.kind]
