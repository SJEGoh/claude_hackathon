"""MAYDAY fast path: a deterministic emergency plan in milliseconds, no negotiation, no LLM in the decision.

When an emergency is declared the normal 3-turn negotiation is abandoned. Code decides immediately:
  - landing: the EARLIEST feasible slot (no added holding) on a runway that is open and inside wind limits
    (TECHNICAL prefers the longest runway). Other traffic in the way is re-sequenced by ATC; wake
    separation with them is ATC's job to restore, not a reason to delay the emergency aircraft.
  - stand: TECHNICAL -> remote stand with fire & rescue; MEDICAL/FUEL -> nearest contact stand free at
    in-block, else remote.
  - if no runway is usable before fuel runs out: divert to the alternate.
Each specialist gets its checklist at the same moment. Agents may then read back their actions (optional,
in parallel); the plan does not wait for them.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from pydantic import BaseModel

from constraints import Proposal
from scenario import WAKE_SEPARATION_MIN, Scenario

EmergencyKind = Literal["FUEL", "MEDICAL", "TECHNICAL"]
KIND_LABEL = {"FUEL": "MAYDAY FUEL", "MEDICAL": "MEDICAL EMERGENCY", "TECHNICAL": "TECHNICAL FAILURE"}


class EmergencyPlan(BaseModel):
    kind: EmergencyKind
    reason: str
    flight: str
    proposal: Optional[Proposal]
    divert_to: Optional[str] = None
    landing_clock: str = "--:--"
    in_block_clock: str = "--:--"
    efob_kg: Optional[int] = None
    fuel_status: str
    displaced: list[str]
    actions: dict[str, list[str]]  # ATC / PILOT / GROUND checklists
    decision_ms: float


def plan_emergency(s: Scenario, kind: EmergencyKind, reason: str = "") -> EmergencyPlan:
    t0 = time.perf_counter()
    usable = [r for r in s.runways if r.wind_within_limits]
    if kind == "TECHNICAL":
        usable.sort(key=lambda r: (-r.length_ft, max(r.closed_until, s.eta_min)))
    else:
        usable.sort(key=lambda r: (max(r.closed_until, s.eta_min), -r.length_ft))
    rwy = usable[0] if usable else None
    slot = max(rwy.closed_until, s.eta_min) if rwy else None

    if rwy is None or s.fuel_at(slot) <= 0:
        actions = {
            "ATC": [f"Coordinate diversion to {s.alternate}; hand off with priority", "Clear the route, no holding"],
            "PILOT": [f"Divert to {s.alternate} now", "Declare MAYDAY on the handoff frequency"],
            "GROUND": [f"Notify {s.alternate} operations, passengers and the next-leg team",
                       "Release the planned stand"],
        }
        return EmergencyPlan(kind=kind, reason=reason, flight=s.flight, proposal=None, divert_to=s.alternate,
                             fuel_status="DIVERT", displaced=[], actions=actions,
                             decision_ms=round((time.perf_counter() - t0) * 1000, 2))

    efob = s.fuel_at(slot)
    fuel_status = ("MAYDAY FUEL (below final reserve)" if efob < s.final_reserve_kg else
                   "MINIMUM FUEL (below FRF + ALTN)" if efob < s.min_landing_fuel_kg else "fuel OK")
    window = max(WAKE_SEPARATION_MIN.values()) + 1
    displaced = [f"{t.callsign} ({s.clock(t.t)})" for t in rwy.traffic if abs(t.t - slot) < window]

    if kind == "TECHNICAL":
        gate = s.gate("R1")
    else:
        contact = [g for g in s.gates if not g.remote and (g.widebody or s.wake != "HEAVY")
                   and g.free_from <= slot + rwy.taxi_min[g.terminal]]
        gate = min(contact, key=lambda g: rwy.taxi_min[g.terminal]) if contact else s.gate("R1")
    on_block = slot + rwy.taxi_min[gate.terminal]
    land, inblk = s.clock(slot), s.clock(on_block)

    atc = [f"PRIORITY: {s.flight} lands RWY {rwy.id} at {land}, no holding",
           (f"Re-sequence {', '.join(displaced)}: extend downwind or hold" if displaced
            else "No conflicting arrivals in the window"),
           f"Stop departures on RWY {rwy.id} from {s.clock(slot - 5)} until the runway is inspected",
           f"Direct taxi route to stand {gate.id}; hold crossing traffic"]
    pilot = [f"Declare {KIND_LABEL[kind]}" + (f": {reason}" if reason else ""),
             f"Expect RWY {rwy.id}, direct approach, touchdown {land}",
             f"EFOB at touchdown {efob} kg: {fuel_status}"]
    ground = [f"Clear stand {gate.id} now; marshaller and tug ready by {inblk}",
              "Hold the ramp team past shift end under the emergency exemption; log the overtime"]
    if kind == "TECHNICAL":
        pilot.append("Run the ECAM/QRH procedure; brief cabin crew for a possible evacuation")
        ground.insert(0, f"ARFF (fire & rescue): position at RWY {rwy.id} by {s.clock(slot - 3)} and follow to {gate.id}")
    elif kind == "MEDICAL":
        pilot.append(f"Cabin crew: prepare the patient for handover at {gate.id}; keep other passengers seated")
        ground.insert(0, f"Paramedics and ambulance at stand {gate.id} by {inblk}; jet bridge first")
    else:
        pilot.append("Fly a minimum-drag, low-power approach; no go-around unless unsafe")
        ground.insert(0, f"ARFF on standby at RWY {rwy.id} from {s.clock(slot - 3)}")
    if s.next_leg:
        ground.append(f"Next leg {s.next_leg.flight}: start an aircraft swap or rebooking now")

    return EmergencyPlan(kind=kind, reason=reason, flight=s.flight,
                         proposal=Proposal(landing_slot_min=slot, runway=rwy.id, gate=gate.id),
                         landing_clock=land, in_block_clock=inblk, efob_kg=efob, fuel_status=fuel_status,
                         displaced=displaced, actions={"ATC": atc, "PILOT": pilot, "GROUND": ground},
                         decision_ms=round((time.perf_counter() - t0) * 1000, 2))


def readbacks(plan: EmergencyPlan, agents: dict, timeout_s: float = 30.0) -> dict[str, str]:
    """Optional: each specialist reads back its checklist (parallel, one short call each). The plan never waits
    for this; any failure just means no readback from that agent."""
    def one(key: str) -> str:
        agent = agents.get(key)
        if agent is None or agent.client is None:
            return ""
        try:
            r = agent.client.with_options(timeout=timeout_s).messages.create(
                model=agent.model, max_tokens=400, output_config={"effort": "low"},
                system=f"{agent.role}\nEmergency in progress: reply with a one-line readback, standard phraseology.",
                messages=[{"role": "user", "content": f"{KIND_LABEL[plan.kind]} {plan.flight}. Your actions:\n- "
                           + "\n- ".join(plan.actions[key]) + "\nRead back."}])
            return "".join(b.text for b in r.content if b.type == "text").strip()
        except Exception:  # noqa: BLE001 - a missing readback must never block the emergency
            return ""
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(zip(("ATC", "PILOT", "GROUND"), pool.map(one, ("ATC", "PILOT", "GROUND"))))
