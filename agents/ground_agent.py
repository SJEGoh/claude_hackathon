"""AOCC agent: Airport Operations Control Centre turnaround coordinator (A-CDM milestones)."""
from __future__ import annotations

from agents.base import BaseAgent
import ops_settings
from constraints import Domain
from scenario import LIMITS, MODELLED_STANDS, Scenario


class GroundAgent(BaseAgent):
    domain: Domain = "GROUND"
    title = "AOCC: Turnaround Coordinator"
    role = ("You are the AOCC turnaround coordinator. Mandate under A-CDM: stand allocation (size and "
            "availability for the whole turnaround), in-block time (EIBT), the next leg's Target Off-Block Time "
            "(TOBT), gate-change notice, ground-crew overtime, and ramp safety (handling stops during lightning). "
            "Minimise turnaround delay and knock-on delay to the next departure.")

    def private_brief(self, s: Scenario) -> str:
        stands = "; ".join(
            f"{g.id} (T{g.terminal}, {'wide' if g.widebody else 'narrow'}{', REMOTE' if g.remote else ''}) free "
            f"{'now' if g.free_from <= -999 else f'{s.clock(g.free_from)} (slot {g.free_from})'} until "
            f"{'open' if g.free_until >= 10_000 else f'{s.clock(g.free_until)} (slot {g.free_until})'}"
            for g in s.gates)
        nl = s.next_leg
        next_leg = (f"Next leg {nl.flight} to {nl.destination}: TOBT {s.clock(nl.tobt)} (slot {nl.tobt})"
                    + (" [DOT schedule]" if nl.source == "DOT" else " [simulated]")) if nl else "No next leg."
        cap = s.airport_limits
        earliest_change = s.decision_time_min + LIMITS.gate_change_notice_min
        return "\n".join([
            f"{s.destination} is a {cap.size_class}: {cap.gates} gates ({min(cap.gates, MODELLED_STANDS)} modelled plus "
            f"remote bay R1), {cap.spare_gates} spare at any time, minimum turnaround {cap.min_turnaround_min} min.",
            f"{s.flight} {s.aircraft_type}: needs a {'wide-body' if s.wake == 'HEAVY' else 'any'} stand. "
            f"Scheduled stand {s.scheduled_gate}. SIBT {s.clock(0)}.",
            f"Turnaround {s.turnaround_min} min from in-block"
            + (" (includes de-icing: freezing precipitation/snow)." if s.deice_required else "."),
            ops_settings.ground_note(s.turnaround_min),
            next_leg,
            f"Gate change: at least {LIMITS.gate_change_notice_min} min notice before arrival (hard). The negotiation "
            f"opened at {s.clock(s.decision_time_min)}, so any stand other than {s.scheduled_gate} needs TLDT "
            f"{s.clock(earliest_change)} (slot {earliest_change}) or later.",
            f"{LIMITS.ground_shift_h} h crew shift ends {s.clock(s.crew_shift_end_min)} (slot {s.crew_shift_end_min}), "
            f"then the next shift takes over; a turnaround running past a shift change is overtime at "
            f"{LIMITS.overtime_multiplier}x (a cost, not a hard limit).",
            (f"RAMP CLOSED (lightning) until {s.clock(s.ramp_closed_until)} (slot {s.ramp_closed_until}): "
             "turnaround cannot start before." if s.ramp_closed_until > -999 else "Ramp open."),
            "Taxi-in by runway -> terminal (min); EIBT = TLDT + taxi: "
            + "; ".join(f"{r.id}: A {r.taxi_min['A']}, B {r.taxi_min['B']}, remote {r.taxi_min['R']}"
                        for r in s.runways),
            "Stand availability: " + stands,
        ])
