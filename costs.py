"""Global cost model: every soft trade-off priced in US dollars, applied ONLY among plans that already pass
every hard safety check (lexicographic: safety first, then cost). Safety is never put on the scale.

Weights are approximate public 2015-era figures, meant to be replaced with an airline's own numbers
(edit in the Control room or assets/cost_weights.json):
  fuel_usd_per_kg            2015 US jet fuel ~ $1.80/US gal (EIA averages); 1 gal Jet-A ~ 3.03 kg -> ~$0.60/kg
  aircraft_time_usd_per_min  airline cost of block time excluding fuel (crew, maintenance, ownership);
                             A4A puts total block-minute cost at ~ $74, of which fuel is roughly a third
  delay_usd_per_min          total cost of a minute of delay to a departure (~ A4A block-minute cost)
  propagation                share of a late in-block that reaches the next departure: 0.76, measured in
                             this project's EDA (inbound arrival delay vs next-leg departure delay)
  missed_connection_usd_per_pax  rebooking, meals, care; (no EU261 in the US) ~ $200
  ground_overtime_usd_per_min    8-person ramp team x ~$30/h x 1.5 overtime ~ $6/min
  remote_stand_usd           buses + extra handling per turnaround ~ $250
  runway_pressure_usd_per_movement  proxy for sector throughput: each movement within +/-15 min of the slot
  rough_hold_usd_per_min     passenger comfort / injury-risk proxy for holding in MOD turbulence or icing
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel

WEIGHTS_FILE = Path(__file__).parent / "assets" / "cost_weights.json"


class Weights(BaseModel):
    fuel_usd_per_kg: float = 0.60
    aircraft_time_usd_per_min: float = 45.0
    delay_usd_per_min: float = 74.0
    propagation: float = 0.76
    missed_connection_usd_per_pax: float = 200.0
    ground_overtime_usd_per_min: float = 6.0
    remote_stand_usd: float = 250.0
    runway_pressure_usd_per_movement: float = 30.0
    rough_hold_usd_per_min: float = 20.0


_CACHE: dict = {}


def weights() -> Weights:
    mtime = WEIGHTS_FILE.stat().st_mtime if WEIGHTS_FILE.exists() else 0
    if _CACHE.get("mtime") != mtime:
        _CACHE.update(mtime=mtime, w=Weights(**json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))) if mtime else Weights())
    return _CACHE["w"]


def save_weights(w: Weights) -> None:
    WEIGHTS_FILE.parent.mkdir(exist_ok=True)
    WEIGHTS_FILE.write_text(w.model_dump_json(indent=1), encoding="utf-8")


def breakdown(pilot: dict, ground: dict, atc: dict, remote: bool, w: Weights | None = None) -> dict[str, float]:
    """Itemised USD cost of a plan from the validators' metrics."""
    w = w or weights()
    rough = pilot.get("holding_min", 0) if "MOD" in (pilot.get("hold_turbulence"), pilot.get("hold_icing")) else 0
    return {
        "PILOT: holding fuel": pilot.get("holding_fuel_kg", 0) * w.fuel_usd_per_kg,
        "PILOT: aircraft time": pilot.get("holding_min", 0) * w.aircraft_time_usd_per_min,
        "PILOT: missed connections": pilot.get("missed_connection_pax", 0) * w.missed_connection_usd_per_pax,
        "PILOT: rough-air holding": rough * w.rough_hold_usd_per_min,
        "GROUND: crew overtime": ground.get("crew_overtime_min", 0) * w.ground_overtime_usd_per_min,
        "GROUND: next-leg knock-on": ground.get("next_leg_delay_min", 0) * w.delay_usd_per_min * w.propagation,
        "GROUND: remote stand": w.remote_stand_usd if remote else 0.0,
        "ATC: runway pressure": atc.get("runway_movements_within_15min", 0) * w.runway_pressure_usd_per_movement,
    }


def robustness(options: dict[str, dict], chosen: str, samples: int = 300, spread: float = 0.5, seed: int = 0) -> float:
    """Share of random weight scenarios (each weight x U[1-spread, 1+spread]) in which `chosen` is still the
    cheapest of `options` ({label: {"pilot":..., "ground":..., "atc":..., "remote":...}})."""
    if len(options) < 2:
        return 1.0
    rng, base, wins = random.Random(seed), weights().model_dump(), 0
    for _ in range(samples):
        w = Weights(**{k: v * rng.uniform(1 - spread, 1 + spread) for k, v in base.items()})
        totals = {k: sum(breakdown(o["pilot"], o["ground"], o["atc"], o["remote"], w).values())
                  for k, o in options.items()}
        wins += min(totals, key=totals.get) == chosen
    return wins / samples
