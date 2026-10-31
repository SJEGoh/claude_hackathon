"""AOCC agent: Airport Operations Control Centre turnaround coordinator (A-CDM milestones)."""
from __future__ import annotations

from agents.base import BaseAgent
from constraints import Domain
import ops_settings
from constraints import turnaround
from scenario import Scenario


class GroundAgent(BaseAgent):
    domain: Domain = "GROUND"
    title = "AOCC: Turnaround Coordinator"
    role = ("You are the AOCC turnaround coordinator. Mandate under A-CDM: stand allocation (size and "
            "availability for the whole turnaround), in-block time (EIBT), the next leg's Target Off-Block Time "
            "(TOBT), ground-crew duty limits and overtime, and ramp safety (handling stops during lightning). "
            "Minimise turnaround delay and knock-on delay to the next departure.")

    def private_brief(self, s: Scenario) -> str:
        stands = "; ".join(
            f"{g.id} (T{g.terminal}, {'wide' if g.widebody else 'narrow'}{', REMOTE' if g.remote else ''}) free "
            f"{'now' if g.free_from <= -999 else f'{s.clock(g.free_from)} (slot {g.free_from})'} until "
            f"{'open' if g.free_until >= 10_000 else f'{s.clock(g.free_until)} (slot {g.free_until})'}"
            for g in s.gates)
        nl, cfg = s.next_leg, ops_settings.get()
        next_leg = (f"Next leg {nl.flight} to {nl.destination}: TOBT {s.clock(nl.tobt)} (slot {nl.tobt})"
                    + (" [DOT schedule]" if nl.source == "DOT" else " [simulated]")) if nl else "No next leg."
        return "\n".join([
            f"{s.flight} {s.aircraft_type}: needs a {'wide-body' if s.wake == 'HEAVY' else 'any'} stand. "
            f"Scheduled stand {s.scheduled_gate}. SIBT {s.clock(0)}.",
            f"Turnaround {turnaround(s)} min from in-block (ramp staffing {cfg.ramp_staffing_pct}%)"
            + (" (includes de-icing: freezing precipitation/snow)." if s.deice_required else "."),
            next_leg,
            f"Crew shift ends {s.clock(s.crew_shift_end_min)} (slot {s.crew_shift_end_min}); overtime allowed up to "
            f"{cfg.max_crew_overtime_min} min (legal cap, hard) but costs money."
            + ("" if cfg.allow_remote_stands else " Remote stands are DISABLED by policy (hard)."),
            (f"RAMP CLOSED (lightning) until {s.clock(s.ramp_closed_until)} (slot {s.ramp_closed_until}): "
             "turnaround cannot start before." if s.ramp_closed_until > -999 else "Ramp open."),
            "Taxi-in by runway -> terminal (min); EIBT = TLDT + taxi: "
            + "; ".join(f"{r.id}: A {r.taxi_min['A']}, B {r.taxi_min['B']}, remote {r.taxi_min['R']}"
                        for r in s.runways),
            "Stand availability: " + stands,
        ])
