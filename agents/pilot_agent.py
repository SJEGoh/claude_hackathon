"""Flight Crew agent: the inbound aircraft's commander, reading FMS fuel predictions and PIREPs."""
from __future__ import annotations

from agents.base import BaseAgent
from constraints import Domain
from scenario import MIN_CONNECT_TIME_MIN, MIN_FUEL_ADVISORY_MARGIN_MIN, SEVERE_HOLD_LIMIT_MIN, Scenario


class PilotAgent(BaseAgent):
    domain: Domain = "PILOT"
    title = "Flight Crew: Captain / FMS"
    role = ("You are the Captain of the inbound aircraft, working from FMS fuel predictions and the latest "
            "PIREPs. Mandate: never land below Final Reserve Fuel plus Alternate fuel (FRF+ALTN); declare "
            "MINIMUM FUEL inside the advisory margin and MAYDAY FUEL below FRF (ICAO); avoid prolonged holding "
            "in reported turbulence or icing; prefer the runway with less crosswind; protect passenger "
            "connections when it costs nothing in safety.")

    def private_brief(self, s: Scenario) -> str:
        wx = s.weather
        pireps = "; ".join(f"{p.raw[:80]} ({p.distance_nm:.0f} NM)" for p in (wx.pireps if wx else [])[:5])
        latest = s.latest_safe_landing
        return "\n".join([
            f"{s.flight} {s.aircraft_type} (wake {s.wake}), tail {s.tail_number}, {s.origin}-{s.destination} "
            f"{s.distance_mi} mi. Inbound delay {s.late_aircraft_delay_min} min late aircraft"
            f"{'' if s.weather_at_destination else f' + {s.weather_delay_min} min en-route weather'}.",
            f"ELDT {s.clock(s.eta_min)} (slot {s.eta_min}). FOB at ELDT {s.fob_at_eta_kg} kg. Holding burn "
            f"{s.hold_burn_kg_per_min} kg/min.",
            f"FRF {s.final_reserve_kg} kg + ALTN ({s.alternate}) {s.alternate_fuel_kg} kg = minimum landing fuel "
            f"{s.min_landing_fuel_kg} kg -> latest safe TLDT {s.clock(latest)} (slot {latest}). MINIMUM FUEL "
            f"advisory inside {MIN_FUEL_ADVISORY_MARGIN_MIN} min of that (slot {latest - MIN_FUEL_ADVISORY_MARGIN_MIN}).",
            f"Destination: {s.weather_attribution}",
            f"PIREPs at holding altitudes (<= FL180, 100 NM, +/-3 h): "
            f"{pireps or 'none filed (absence is not proof of smooth air)'}."
            + (f" Max {SEVERE_HOLD_LIMIT_MIN} min holding in reported SEV conditions."
               if wx and "SEV" in (wx.hold_turbulence, wx.hold_icing) else ""),
            "Runway crosswind components: " + ", ".join(f"{r.id} {r.crosswind_kt} kt" for r in s.runways),
            f"Onward connections (MCT {MIN_CONNECT_TIME_MIN} min after in-block): "
            + (", ".join(f"{c.flight} to {c.destination} {c.pax} pax departs {s.clock(c.departs_at)} "
                         f"(slot {c.departs_at})" for c in s.connections) or "none tracked"),
        ])
