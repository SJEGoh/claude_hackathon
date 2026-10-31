"""ATC agent: arrival manager (AMAN) + ground movement control. Owns the landing sequence."""
from __future__ import annotations

from typing import Optional

from agents.base import BaseAgent
from constraints import Domain, Proposal, check_atc
import ops_settings
from scenario import WAKE_SEPARATION_MIN, Scenario


class ATCAgent(BaseAgent):
    domain: Domain = "ATC"
    title = "ATC: Arrival Manager / Ground Movement"
    role = ("You are the approach controller running the Arrival Manager (AMAN) and ground movement at the "
            "destination. Mandate: wake-turbulence separation, runway-in-use selection within crosswind and "
            "tailwind limits, runway closures, taxiway restrictions, and maximum arrival throughput. You own "
            "the landing sequence: you table the plan each turn and fold the Flight Crew's and AOCC's "
            "published constraints into it.")

    def private_brief(self, s: Scenario) -> str:
        cfg = ops_settings.get()
        seps = ", ".join(f"{a[0]}->{b[0]} {v}" for (a, b), v in WAKE_SEPARATION_MIN.items())
        wx = s.weather
        lines = [
            f"Inbound {s.flight} ({s.aircraft_type}, wake {s.wake}) {s.origin}-{s.destination}. "
            f"ELDT {s.clock(s.eta_min)} (slot {s.eta_min}). SIBT {s.clock(0)}.",
            f"METAR: {wx.metar if wx and wx.metar else 'not available'}",
            f"Conditions: {'IMC' if s.imc else 'VMC'}; approach mode {cfg.approach}; runway condition RWYCC {cfg.rwycc}"
            f" (+{cfg.extra_spacing(s.imc)} min spacing on every pair). Wake minima min (leader->follower): {seps}. "
            f"Limits: crosswind {cfg.crosswind_limit():.0f} kt, tailwind {cfg.max_tailwind_kt:.0f} kt"
            + (". RWYCC 0: ALL RUNWAYS CLOSED (nil braking)." if cfg.runway_closed_by_condition() else "."),
            s.taxiway_note,
            f"Scheduled stand on the flight plan: {s.scheduled_gate} (AOCC owns stand allocation).",
        ]
        for r in s.runways:
            window = [t for t in r.traffic if s.eta_min - 5 <= t.t <= s.eta_min + 75]
            lines.append(
                f"RWY {r.id} (hdg {r.heading_true:.0f}T, {r.length_ft} ft): {r.closure_reason}"
                + (f" until {s.clock(r.closed_until)} (slot {r.closed_until})" if r.closed_until > -999 else "")
                + f"; wind head {r.headwind_kt} / cross {r.crosswind_kt} / tail {r.tailwind_kt} kt"
                + f"; taxi-in to terminal A {r.taxi_min['A']}, B {r.taxi_min['B']}, remote {r.taxi_min['R']} min"
                + "; arrivals: " + ", ".join(f"{t.t}({t.wake[0]})" for t in window))
        lines.append("AMAN conflict-free slots (ATC checks only; fuel, stands and crew NOT considered): "
                     + "; ".join(f"{rid}: {v}" for rid, v in self.free_slots(s).items()))
        return "\n".join(lines)

    @staticmethod
    def free_slots(s: Scenario, per_runway: int = 8) -> dict[str, list[int]]:
        out = {}
        for r in s.runways:
            found = []
            for t in range(s.eta_min, s.eta_min + 180):
                if check_atc(s, Proposal(landing_slot_min=t, runway=r.id, gate=s.scheduled_gate)).hard_ok:
                    found.append(t)
                    if len(found) == per_runway:
                        break
            out[r.id] = found
        return out

    def task(self, turn: int, on_table: Optional[Proposal], plan_id: int) -> str:
        if turn == 1:
            return ("Stance PROPOSE, answering_plan=0. Table the first plan: an AMAN conflict-free slot and "
                    "runway. Keep the scheduled stand unless you know it is unusable; AOCC confirms or counters.")
        return ("Stance PROPOSE, answering_plan=0. Table a revised plan that satisfies every HARD constraint "
                "Flight Crew and AOCC have published and keeps separation. If a counter-proposal or the Chief "
                "of Staff's bridge plan works for you, adopt it unchanged. Re-tabling a plan that was already "
                "countered without change is treated as a deadlock.")
