"""Deterministic (non-AI) constraint engine.

The LLM agents negotiate, but they never get the last word on safety: every plan is checked here in plain
code, and a plan that fails any HARD check can never be agreed. This module also holds the deterministic
planner used as the non-AI fallback and to list safe alternatives for a human decision.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

import costs
from scenario import (
    IMC_EXTRA_SPACING_MIN, MAX_CREW_OVERTIME_MIN, MIN_CONNECT_TIME_MIN, MIN_FUEL_ADVISORY_MARGIN_MIN,
    SEVERE_HOLD_LIMIT_MIN, WAKE_SEPARATION_MIN, Scenario,
)
from weather import MAX_CROSSWIND_KT, MAX_TAILWIND_KT

Domain = Literal["ATC", "PILOT", "GROUND"]


class Proposal(BaseModel):
    landing_slot_min: int  # minutes relative to SIBT
    runway: str
    gate: str


class Check(BaseModel):
    domain: Domain
    name: str
    passed: bool
    hard: bool
    detail: str


class DomainReport(BaseModel):
    domain: Domain
    checks: list[Check]
    metrics: dict[str, float | int | str]

    @property
    def hard_ok(self) -> bool:
        return all(c.passed for c in self.checks if c.hard)

    def violations(self) -> list[Check]:
        return [c for c in self.checks if c.hard and not c.passed]

    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.hard and not c.passed]


def on_block(s: Scenario, p: Proposal) -> Optional[int]:
    gate, rwy = s.gate(p.gate), s.runway(p.runway)
    if gate is None or rwy is None:
        return None
    return p.landing_slot_min + rwy.taxi_min[gate.terminal]


# ============================================================================
# Domain validators
# ============================================================================

def check_atc(s: Scenario, p: Proposal) -> DomainReport:
    rwy = s.runway(p.runway)
    if rwy is None:
        return DomainReport(domain="ATC", metrics={}, checks=[Check(
            domain="ATC", name="runway_exists", passed=False, hard=True, detail=f"Runway {p.runway} not in use")])
    checks = [
        Check(domain="ATC", name="aircraft_can_reach_slot", hard=True, passed=p.landing_slot_min >= s.eta_min,
              detail=f"TLDT {s.clock(p.landing_slot_min)} vs ELDT {s.clock(s.eta_min)}"),
        Check(domain="ATC", name="runway_open", hard=True, passed=p.landing_slot_min >= rwy.closed_until,
              detail=f"{rwy.id} {rwy.closure_reason}" + (f" until {s.clock(rwy.closed_until)}"
                                                         if rwy.closed_until > -999 else "")),
        Check(domain="ATC", name="runway_wind_limits", hard=True, passed=rwy.wind_within_limits,
              detail=f"{rwy.id} crosswind {rwy.crosswind_kt} kt (max {MAX_CROSSWIND_KT}), tailwind "
                     f"{rwy.tailwind_kt} kt (max {MAX_TAILWIND_KT})"),
    ]
    extra = IMC_EXTRA_SPACING_MIN if s.imc else 0
    conflicts = []
    for other in rwy.traffic:
        if other.t <= p.landing_slot_min:
            need, gap = WAKE_SEPARATION_MIN[(other.wake, s.wake)] + extra, p.landing_slot_min - other.t
        else:
            need, gap = WAKE_SEPARATION_MIN[(s.wake, other.wake)] + extra, other.t - p.landing_slot_min
        if gap < need:
            conflicts.append(f"{other.callsign}({other.wake[0]}) at {s.clock(other.t)}: {gap} < {need} min")
    checks.append(Check(domain="ATC", name="wake_separation", hard=True, passed=not conflicts,
                        detail="; ".join(conflicts) or f"all gaps meet wake minima{' (+1 min IMC)' if extra else ''}"))
    gate = s.gate(p.gate)
    if gate is not None:
        taxi = rwy.taxi_min[gate.terminal]
        checks.append(Check(domain="ATC", name="taxi_route", hard=False, passed=taxi <= 12,
                            detail=f"{rwy.id} -> Terminal {gate.terminal}: {taxi} min"))
    load = sum(1 for t in rwy.traffic if abs(t.t - p.landing_slot_min) <= 15)
    return DomainReport(domain="ATC", checks=checks, metrics={"runway_movements_within_15min": load})


def check_pilot(s: Scenario, p: Proposal) -> DomainReport:
    holding = max(0, p.landing_slot_min - s.eta_min)
    efob = s.fuel_at(p.landing_slot_min)
    margin_min = (efob - s.min_landing_fuel_kg) / s.hold_burn_kg_per_min
    checks = [
        Check(domain="PILOT", name="fuel_reserve", hard=True, passed=efob >= s.min_landing_fuel_kg,
              detail=f"EFOB {efob} kg vs FRF+ALTN {s.min_landing_fuel_kg} kg ({margin_min:+.0f} min)"),
        Check(domain="PILOT", name="minimum_fuel_advisory", hard=False,
              passed=margin_min >= MIN_FUEL_ADVISORY_MARGIN_MIN,
              detail=f"{margin_min:.0f} min above FRF+ALTN (MINIMUM FUEL inside {MIN_FUEL_ADVISORY_MARGIN_MIN})"),
    ]
    wx = s.weather
    turb, ice = (wx.hold_turbulence, wx.hold_icing) if wx else ("NONE", "NONE")
    severe = "SEV" in (turb, ice)
    checks.append(Check(domain="PILOT", name="holding_environment", hard=severe,
                        passed=(holding <= SEVERE_HOLD_LIMIT_MIN) if severe else (holding == 0 or "MOD" not in (turb, ice)),
                        detail=f"{holding} min holding; PIREPs turbulence {turb}, icing {ice}"
                               + (f" (max {SEVERE_HOLD_LIMIT_MIN} min in SEV)" if severe else "")))
    rwy = s.runway(p.runway)
    if rwy:
        checks.append(Check(domain="PILOT", name="approach_crosswind", hard=False, passed=rwy.crosswind_kt <= 20,
                            detail=f"{rwy.id} crosswind {rwy.crosswind_kt} kt"))
    ob = on_block(s, p)
    missed = [c for c in s.connections if ob is None or ob + MIN_CONNECT_TIME_MIN > c.departs_at]
    checks.append(Check(domain="PILOT", name="passenger_connections", hard=False, passed=not missed,
                        detail=", ".join(f"{c.flight} to {c.destination} ({c.pax} pax)" for c in missed)
                        or (f"{len(s.connections)} connections protected" if s.connections else "no tracked connections")))
    return DomainReport(domain="PILOT", checks=checks, metrics={
        "holding_min": holding, "efob_kg": efob, "fuel_margin_min": round(margin_min, 1),
        "holding_fuel_kg": holding * s.hold_burn_kg_per_min,
        "missed_connection_pax": sum(c.pax for c in missed), "hold_turbulence": turb, "hold_icing": ice,
    })


def check_ground(s: Scenario, p: Proposal) -> DomainReport:
    gate = s.gate(p.gate)
    ob = on_block(s, p)
    if gate is None or ob is None:
        return DomainReport(domain="GROUND", metrics={}, checks=[Check(
            domain="GROUND", name="gate_exists", passed=False, hard=True, detail=f"Stand {p.gate} unknown")])
    start = max(ob, s.ramp_closed_until)
    end = start + s.turnaround_min
    overtime = max(0, end - s.crew_shift_end_min)
    tobt_delay = max(0, end - s.next_leg.tobt) if s.next_leg else 0
    checks = [
        Check(domain="GROUND", name="stand_size", hard=True, passed=gate.widebody or s.wake != "HEAVY",
              detail=f"{gate.id} {'wide' if gate.widebody else 'narrow'}-body stand, aircraft {s.aircraft_type}"),
        Check(domain="GROUND", name="stand_free_at_aibt", hard=True, passed=gate.free_from <= ob,
              detail=f"EIBT {s.clock(ob)}, {gate.id} free from {s.clock(gate.free_from) if gate.free_from > -999 else 'now'}"),
        Check(domain="GROUND", name="stand_free_for_turnaround", hard=True, passed=end <= gate.free_until,
              detail=f"turnaround ends {s.clock(end)}, next user of {gate.id} at "
                     f"{s.clock(gate.free_until) if gate.free_until < 10_000 else 'none'}"),
        Check(domain="GROUND", name="crew_duty_limit", hard=True, passed=overtime <= MAX_CREW_OVERTIME_MIN,
              detail=f"overtime {overtime} min (legal cap {MAX_CREW_OVERTIME_MIN})"),
        Check(domain="GROUND", name="crew_overtime", hard=False, passed=overtime == 0,
              detail=f"shift ends {s.clock(s.crew_shift_end_min)}, turnaround ends {s.clock(end)}"),
        Check(domain="GROUND", name="next_leg_tobt", hard=False, passed=tobt_delay == 0,
              detail=(f"{s.next_leg.flight} TOBT {s.clock(s.next_leg.tobt)}: {tobt_delay} min knock-on delay"
                      if s.next_leg else "no next leg")),
        Check(domain="GROUND", name="contact_stand", hard=False, passed=not gate.remote,
              detail="remote stand: bussing" if gate.remote else "contact stand"),
    ]
    if s.ramp_closed_until > -999:
        checks.append(Check(domain="GROUND", name="ramp_lightning_hold", hard=False, passed=ob >= s.ramp_closed_until,
                            detail=f"ramp closed (lightning) until {s.clock(s.ramp_closed_until)}"))
    return DomainReport(domain="GROUND", checks=checks, metrics={
        "eibt_min": ob, "turnaround_end_min": end, "crew_overtime_min": overtime, "next_leg_delay_min": tobt_delay,
    })


VALIDATORS = {"ATC": check_atc, "PILOT": check_pilot, "GROUND": check_ground}


def check_all(s: Scenario, p: Proposal) -> dict[Domain, DomainReport]:
    return {d: fn(s, p) for d, fn in VALIDATORS.items()}


def is_safe(reports: dict[Domain, DomainReport]) -> bool:
    return all(r.hard_ok for r in reports.values())


def cost_breakdown(s: Scenario, p: Proposal, reports: Optional[dict[Domain, DomainReport]] = None) -> dict[str, float]:
    """Itemised USD cost (costs.py). Only meaningful for plans that pass every hard check."""
    r = reports or check_all(s, p)
    gate = s.gate(p.gate)
    return costs.breakdown(r["PILOT"].metrics, r["GROUND"].metrics, r["ATC"].metrics, bool(gate and gate.remote))


def cost(s: Scenario, p: Proposal, reports: Optional[dict[Domain, DomainReport]] = None) -> float:
    """Total USD cost of the soft trade-offs (lexicographic: rank safe plans only)."""
    return round(sum(cost_breakdown(s, p, reports).values()), 2)


def plan_robustness(s: Scenario, chosen: Proposal, others: list[Proposal]) -> float:
    """How often `chosen` stays cheapest among safe options when every weight is varied +/-50%."""
    opts = {}
    for p in [chosen, *others]:
        r = check_all(s, p)
        if is_safe(r):
            gate = s.gate(p.gate)
            opts[str((p.landing_slot_min, p.runway, p.gate))] = {
                "pilot": r["PILOT"].metrics, "ground": r["GROUND"].metrics, "atc": r["ATC"].metrics,
                "remote": bool(gate and gate.remote)}
    key = str((chosen.landing_slot_min, chosen.runway, chosen.gate))
    return costs.robustness(opts, key) if key in opts else 0.0


# ============================================================================
# Deterministic non-AI planner
# ============================================================================

class FallbackResult(BaseModel):
    proposal: Proposal
    mode: Literal["OPTIMISED_SOP", "FUEL_PRIORITY", "DIVERT"]
    reason: str
    cost: Optional[float]
    alternatives: list[tuple[Proposal, float]] = []  # next-best safe plans, for human choice


def deterministic_plan(s: Scenario, horizon_min: int = 180, k: int = 4) -> FallbackResult:
    """Exhaustive search over slot x runway x gate for the cheapest fully-safe plans.

    If nothing satisfies every hard constraint: fuel priority (ICAO MAYDAY/MINIMUM FUEL handling, ATC
    re-sequences other traffic) onto the earliest usable runway and the remote stand; divert to the
    alternate if even that breaches FRF+ALTN.
    """
    found: list[tuple[float, Proposal]] = []
    for slot in range(s.eta_min, s.eta_min + horizon_min + 1):
        if s.fuel_at(slot) < s.min_landing_fuel_kg:
            break
        for rwy in s.runways:
            for gate in s.gates:
                p = Proposal(landing_slot_min=slot, runway=rwy.id, gate=gate.id)
                reports = check_all(s, p)
                if is_safe(reports):
                    found.append((cost(s, p, reports), p))
    if found:
        found.sort(key=lambda x: (x[0], x[1].landing_slot_min))
        alts, seen = [], {(found[0][1].runway, found[0][1].gate)}
        for c, p in found[1:]:  # diverse alternatives: one per runway/gate pair
            if (p.runway, p.gate) not in seen:
                seen.add((p.runway, p.gate))
                alts.append((p, c))
            if len(alts) == k - 1:
                break
        return FallbackResult(proposal=found[0][1], mode="OPTIMISED_SOP", cost=found[0][0], alternatives=alts,
                              reason="Least-cost plan satisfying every hard constraint (exhaustive search).")
    usable = [r for r in s.runways if r.wind_within_limits] or s.runways
    rwy = min(usable, key=lambda r: max(r.closed_until, s.eta_min))
    slot = max(rwy.closed_until, s.eta_min)
    p = Proposal(landing_slot_min=slot, runway=rwy.id, gate="R1")
    if rwy.wind_within_limits and check_pilot(s, p).hard_ok:
        return FallbackResult(proposal=p, mode="FUEL_PRIORITY", cost=None,
                              reason=f"No conflict-free plan exists. Fuel priority: ATC re-sequences traffic on "
                                     f"{rwy.id}, aircraft parks on remote stand R1.")
    return FallbackResult(proposal=Proposal(landing_slot_min=s.eta_min, runway="DIVERT", gate=s.alternate),
                          mode="DIVERT", cost=None,
                          reason=f"FRF+ALTN cannot be protected at {s.destination}: divert to {s.alternate}.")
