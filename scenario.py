"""Disruption scenarios built from the US DOT 2015 data, real weather and real runways.

Time base: integer minutes relative to the flight's Scheduled In-Block Time (SIBT = DOT SCHEDULED_ARRIVAL,
local time). Landing-slot offsets are in the same base, so an offset can be negative (landing before SIBT).

What is REAL (traceable to a source) and what is SIMULATED (seeded, deterministic per DOT row):
  real       flight, route, date, delays (DOT); arrivals of other DOT flights at that airport (WHEELS_ON);
             real onward connections and the aircraft's next leg (DOT); taxi-in times and turnaround norms
             (EDA calibration); runway identifiers and headings (OurAirports); METAR, PIREPs, ERA5 weather
             (IEM, Open-Meteo, or live aviationweather.gov); nearest plausible alternate (airports.csv)
  simulated  extra background traffic to match the EDA arrival-rate estimate (the extract is ~1/15 of all
             flights), gate layout and occupancy, fuel on board, crew shift end, passenger counts, and the
             injected taxiway blockage
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel

import weather as W

ROOT = Path(__file__).parent
FLIGHTS_CSV = ROOT / "flights.csv"
AIRPORTS_CSV = ROOT / "airports.csv"
CALIBRATION = ROOT / "eda" / "calibration.json"

Wake = Literal["HEAVY", "MEDIUM"]
WeatherMode = Literal["HISTORICAL", "LIVE", "OFF"]

# Minimum time (min) between a LEADER and the FOLLOWER landing behind it on the same runway (simplified
# time-based ICAO wake categories). IMC adds IMC_EXTRA_SPACING_MIN to every pair.
WAKE_SEPARATION_MIN = {("HEAVY", "HEAVY"): 2, ("HEAVY", "MEDIUM"): 3, ("MEDIUM", "HEAVY"): 2,
                       ("MEDIUM", "MEDIUM"): 2}
IMC_EXTRA_SPACING_MIN = 1
MIN_CONNECT_TIME_MIN = 40
MAX_CREW_OVERTIME_MIN = 30
FINAL_RESERVE_MIN = 30          # ICAO final reserve fuel, at holding burn
MIN_FUEL_ADVISORY_MARGIN_MIN = 10  # ICAO "MINIMUM FUEL" is declared inside this margin above FRF + ALTN
SEVERE_HOLD_LIMIT_MIN = 10      # max holding inside reported SEV turbulence / icing
DEICE_EXTRA_MIN = 15
MAX_RUNWAY_RATE_PER_H = {"VMC": 28, "IMC": 22}
WIDEBODY_DISTANCE_MI = 2000
AIRCRAFT = {  # Airbus-family stand-ins chosen by wake category (DOT has no aircraft type)
    "MEDIUM": {"type": "A320neo-class", "hold_kg_min": 38, "cruise_kg_min": 42, "cruise_nm_min": 7.3},
    "HEAVY": {"type": "A330neo-class", "hold_kg_min": 95, "cruise_kg_min": 100, "cruise_nm_min": 7.8},
}


class TrafficSlot(BaseModel):
    t: int
    callsign: str
    wake: Wake
    source: Literal["DOT", "SIM"]


class Runway(BaseModel):
    id: str
    heading_true: float
    length_ft: int
    closed_until: int
    closure_reason: str
    headwind_kt: float
    crosswind_kt: float
    tailwind_kt: float
    wind_within_limits: bool
    taxi_min: dict[str, int]  # terminal -> taxi-in minutes
    traffic: list[TrafficSlot]


class Gate(BaseModel):
    id: str
    terminal: str
    widebody: bool
    free_from: int
    free_until: int
    remote: bool = False


class Connection(BaseModel):
    flight: str
    destination: str
    pax: int
    departs_at: int
    source: Literal["DOT"] = "DOT"


class NextLeg(BaseModel):
    flight: str
    destination: str
    tobt: int  # target off-block time (scheduled departure offset)
    source: Literal["DOT", "SIM"]


class Scenario(BaseModel):
    # --- DOT row ---
    row_index: int
    flight: str
    airline: str
    tail_number: str
    origin: str
    destination: str
    destination_name: str
    date: str
    scheduled_arrival_hhmm: str
    weather_delay_min: int
    late_aircraft_delay_min: int
    distance_mi: int

    # --- context ---
    weather_mode: WeatherMode
    weather: Optional[W.WeatherBrief]
    weather_at_destination: bool
    weather_attribution: str
    imc: bool
    airport_icao: str

    # --- PILOT domain ---
    wake: Wake
    aircraft_type: str
    eta_min: int  # earliest landing (ELDT) given the inbound delay
    fob_at_eta_kg: int
    hold_burn_kg_per_min: int
    final_reserve_kg: int
    alternate: str
    alternate_fuel_kg: int
    connections: list[Connection]

    # --- ATC domain ---
    runways: list[Runway]
    taxiway_note: str

    # --- GROUND domain ---
    gates: list[Gate]
    scheduled_gate: str
    turnaround_min: int
    deice_required: bool
    ramp_closed_until: int  # lightning: ground handling suspended
    crew_shift_end_min: int
    next_leg: Optional[NextLeg]

    @property
    def min_landing_fuel_kg(self) -> int:
        return self.final_reserve_kg + self.alternate_fuel_kg

    @property
    def latest_safe_landing(self) -> int:
        return self.eta_min + (self.fob_at_eta_kg - self.min_landing_fuel_kg) // self.hold_burn_kg_per_min

    def fuel_at(self, slot: int) -> int:
        return self.fob_at_eta_kg - max(0, slot - self.eta_min) * self.hold_burn_kg_per_min

    def clock(self, offset_min: int) -> str:
        hh, mm = int(self.scheduled_arrival_hhmm[:2]) % 24, int(self.scheduled_arrival_hhmm[2:])
        total = (hh * 60 + mm + offset_min) % (24 * 60)
        return f"{total // 60:02d}:{total % 60:02d}"

    def runway(self, rid: str) -> Optional[Runway]:
        return next((r for r in self.runways if r.id == rid), None)

    def gate(self, gid: str) -> Optional[Gate]:
        return next((g for g in self.gates if g.id == gid), None)


# ============================================================================
# Data loading
# ============================================================================

MAX_WEATHER_DELAY = 90
MAX_LATE_AIRCRAFT_DELAY = 120


@lru_cache(maxsize=1)
def all_flights() -> pd.DataFrame:
    df = pd.read_csv(FLIGHTS_CSV, low_memory=False, dtype={"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str})
    df["SA_MIN"] = df.SCHEDULED_ARRIVAL // 100 % 24 * 60 + df.SCHEDULED_ARRIVAL % 100
    df["SD_MIN"] = df.SCHEDULED_DEPARTURE // 100 % 24 * 60 + df.SCHEDULED_DEPARTURE % 100
    df["WO_MIN"] = df.WHEELS_ON // 100 % 24 * 60 + df.WHEELS_ON % 100
    return df


def bucket_of(weather: float, late: float) -> str:
    if weather > 15 and late > 20:
        return "both"
    return "weather" if weather > 15 else "late_aircraft"


def load_disrupted_flights(limit: Optional[int] = 500) -> pd.DataFrame:
    """Rows matching WEATHER_DELAY > 15 or LATE_AIRCRAFT_DELAY > 20 that a tactical negotiation can act on."""
    df = all_flights()
    df = df[(df.CANCELLED == 0) & (df.DIVERTED == 0)]
    df = df[(df.WEATHER_DELAY > 15) | (df.LATE_AIRCRAFT_DELAY > 20)]
    df = df[(df.WEATHER_DELAY <= MAX_WEATHER_DELAY) & (df.LATE_AIRCRAFT_DELAY <= MAX_LATE_AIRCRAFT_DELAY)]
    df = df.reset_index()
    df["BUCKET"] = [bucket_of(w, l) for w, l in zip(df.WEATHER_DELAY, df.LATE_AIRCRAFT_DELAY)]
    return df if limit is None else df.head(limit)


@lru_cache(maxsize=1)
def calibration() -> dict:
    return json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}


@lru_cache(maxsize=1)
def _airports() -> pd.DataFrame:
    return pd.read_csv(AIRPORTS_CSV)


@lru_cache(maxsize=1)
def _busy_airports() -> set[str]:
    counts = all_flights().DESTINATION_AIRPORT.value_counts()
    return set(counts[counts >= 500].index)


def _nearest_alternate(dest: str) -> tuple[str, float]:
    a = _airports().dropna(subset=["LATITUDE"])
    row = a[a.IATA_CODE == dest]
    if row.empty:
        return "n/a", 150.0
    lat, lon = float(row.LATITUDE.iloc[0]), float(row.LONGITUDE.iloc[0])
    best = None
    for r in a[a.IATA_CODE.isin(_busy_airports()) & (a.IATA_CODE != dest)].itertuples():
        d = W._distance_nm(lat, lon, r.LATITUDE, r.LONGITUDE)
        if d >= 60 and (best is None or d < best[1]):
            best = (r.IATA_CODE, d)
    return best or ("n/a", 150.0)


# ============================================================================
# Scenario synthesis
# ============================================================================

_PREFIX = {"AA": "AAL", "DL": "DAL", "UA": "UAL", "WN": "SWA", "B6": "JBU", "AS": "ASA", "OO": "SKW",
           "EV": "ASQ", "F9": "FFT", "NK": "NKS", "HA": "HAL", "VX": "VRD", "MQ": "ENY", "US": "AWE"}


def _wrap(offset: int) -> int:
    """Normalise a minute difference into (-720, 720]."""
    return (offset + 720) % 1440 - 720


def _pick_runways(ref: Optional[W.AirportRef], brief: Optional[W.WeatherBrief]) -> list[tuple[W.RunwayEnd, W.RunwayWind]]:
    if ref is None or not ref.runway_ends:
        fake = [W.RunwayEnd(ident=i, heading_true=h, length_ft=10000, physical="A/B") for i, h in (("RWY1", 0), ("RWY2", 90))]
        return [(e, W.RunwayWind(runway=e.ident, headwind_kt=0, crosswind_kt=0, tailwind_kt=0, within_limits=True))
                for e in fake]
    winds = {w.runway: w for w in (brief.runway_winds if brief else W.runway_winds(ref.runway_ends, None, None, None))}
    ranked = sorted(ref.runway_ends, key=lambda e: (not winds[e.ident].within_limits, -winds[e.ident].headwind_kt,
                                                     -e.length_ft, e.ident))
    chosen, physicals = [], set()
    for e in ranked:
        if e.physical in physicals:
            continue
        chosen.append((e, winds[e.ident]))
        physicals.add(e.physical)
        if len(chosen) == 2:
            break
    return chosen


def build_scenario(row: pd.Series, weather_mode: WeatherMode = "HISTORICAL") -> Scenario:
    key = f"{row.AIRLINE}{row.FLIGHT_NUMBER}-{row.YEAR}{int(row.MONTH):02d}{int(row.DAY):02d}-{row.TAIL_NUMBER}"
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
    dest = str(row.DESTINATION_AIRPORT)
    cal = calibration()
    cal_ap = cal.get("airports", {}).get(dest, {})
    weather_delay, late = int(row.WEATHER_DELAY), int(row.LATE_AIRCRAFT_DELAY)
    wake: Wake = "HEAVY" if row.DISTANCE >= WIDEBODY_DISTANCE_MI else "MEDIUM"
    ac = AIRCRAFT[wake]
    sibt_min = int(row.SCHEDULED_ARRIVAL) // 100 % 24 * 60 + int(row.SCHEDULED_ARRIVAL) % 100
    sibt_local = datetime(int(row.YEAR), int(row.MONTH), int(row.DAY)) + timedelta(minutes=sibt_min)

    # --- weather (deterministic, from sources) ------------------------------
    ref = W.airport_ref(dest)
    brief: Optional[W.WeatherBrief] = None
    if ref and weather_mode == "HISTORICAL":
        brief = W.historical_brief(ref, sibt_local + timedelta(minutes=late))
    elif ref and weather_mode == "LIVE":
        brief = W.live_brief(ref)
    picked = _pick_runways(ref, brief)
    imc = bool(brief and brief.flight_category in ("IFR", "LIFR"))
    dest_wx = bool(brief and (brief.thunderstorm or brief.convective_nearby or imc or brief.freezing_precip_or_snow
                              or not any(w.within_limits for _, w in picked)))
    if brief is None:
        attribution = ("No weather source available: WEATHER_DELAY treated as origin/en-route weather. "
                       "Agents are told destination weather is UNKNOWN.")
    elif dest_wx or weather_delay == 0:
        attribution = (f"Destination weather at ETA: {brief.summary()}."
                       + (f" DOT attributes {weather_delay} min to weather: arrivals suspended for it."
                          if dest_wx and weather_delay else ""))
    else:
        attribution = (f"Destination is {brief.flight_category} at ETA ({brief.summary()}), so the "
                       f"{weather_delay} min DOT WEATHER_DELAY is attributed to origin/en-route weather and "
                       "is already in the inbound delay.")

    taxi_p50 = int(round(cal_ap.get("taxi_in_p50", 7)))
    taxi_p75 = int(round(cal_ap.get("taxi_in_p75", taxi_p50 + 3)))
    inbound_delay = late + (0 if dest_wx else weather_delay)
    eta = inbound_delay - taxi_p50

    # --- runways, closures, traffic ------------------------------------------
    weather_close = weather_delay - taxi_p50 if (dest_wx and weather_delay) else -999
    reason = ("thunderstorm over the field" if brief and brief.thunderstorm else
              "convective cell on final" if brief and brief.convective_nearby else
              f"{brief.flight_category} conditions, approach suspended" if imc else
              "freezing precipitation, runway treatment" if brief and brief.freezing_precip_or_snow else "")
    flights = all_flights()
    same_day = flights[(flights.DESTINATION_AIRPORT == dest) & (flights.MONTH == row.MONTH) & (flights.DAY == row.DAY)
                       & (flights.index != row.get("index", -1))]
    real = [(_wrap(int(wo) - sibt_min), f"{_PREFIX.get(a, a)}{int(n)}", "HEAVY" if d >= WIDEBODY_DISTANCE_MI else "MEDIUM")
            for wo, a, n, d in zip(same_day.WO_MIN, same_day.AIRLINE, same_day.FLIGHT_NUMBER, same_day.DISTANCE)]
    window = (eta - 20, eta + 160)
    real = sorted(r for r in real if window[0] <= r[0] <= window[1])
    hour = (sibt_min + max(eta, 0)) // 60 % 24
    est_rate = cal_ap.get("arrivals_per_hour", {}).get(str(hour), 20.0)
    cap = MAX_RUNWAY_RATE_PER_H["IMC" if imc else "VMC"]
    per_runway_rate = min(cap, max(8.0, est_rate / max(1, len(picked))))

    runways: list[Runway] = []
    for i, (end, wind) in enumerate(picked):
        closed = weather_close - (rng.randint(4, 9) if i == 1 else 0) if weather_close > -999 else -999
        if not wind.within_limits:
            closure = f"wind out of limits (crosswind {wind.crosswind_kt} kt, tailwind {wind.tailwind_kt} kt)"
        else:
            closure = reason if closed > -999 else "open"
        mine = [TrafficSlot(t=t, callsign=c, wake=w, source="DOT") for j, (t, c, w) in enumerate(real)
                if j % len(picked) == i and t >= closed]
        sim_t, slots = max(window[0], closed), []
        gap = 60 / per_runway_rate
        backlog_until = closed + 30 if closed > -999 else -999
        while sim_t <= window[1]:
            step = 2.2 if sim_t < backlog_until else gap * rng.uniform(0.75, 1.25)
            t = int(round(sim_t))
            if all(abs(t - s.t) >= 2 for s in mine + slots):
                slots.append(TrafficSlot(t=t, callsign=f"{rng.choice(list(_PREFIX.values()))}{rng.randint(100, 3999)}",
                                         wake="HEAVY" if rng.random() < 0.15 else "MEDIUM", source="SIM"))
            sim_t += step
        taxi = {"A": taxi_p50 if i == 0 else taxi_p75 + 12, "B": taxi_p50 + 3 if i == 0 else taxi_p75, "R": taxi_p50 + 4}
        runways.append(Runway(id=end.ident, heading_true=end.heading_true, length_ft=end.length_ft,
                              closed_until=closed if closed > -999 else -999, closure_reason=closure,
                              headwind_kt=wind.headwind_kt, crosswind_kt=wind.crosswind_kt,
                              tailwind_kt=wind.tailwind_kt, wind_within_limits=wind.within_limits,
                              taxi_min=taxi, traffic=sorted(mine + slots, key=lambda s: s.t)))
    taxiway_note = (f"INJECTED DISRUPTION: taxiway K closed (disabled tug). {runways[-1].id} -> Terminal A "
                    "reroutes via the perimeter taxiway, +12 min." if len(runways) > 1 else "No taxiway restrictions.")

    # --- ground ----------------------------------------------------------------
    cal_turn = cal.get("turnaround_min", {"MEDIUM": 40, "HEAVY": 56})
    deice = bool(brief and brief.freezing_precip_or_snow)
    turnaround = int(cal_turn[wake]) + (DEICE_EXTRA_MIN if deice else 0)
    gates: list[Gate] = []
    for terminal in ("A", "B"):
        for n in range(1, 6):
            free_from = rng.choice([-60, -60, eta - 5, eta + 15, eta + 35, eta + 60])
            gates.append(Gate(id=f"{terminal}{n}", terminal=terminal, widebody=n <= 2, free_from=free_from,
                              free_until=free_from + rng.choice([70, 100, 140, 200, 600])))
    gates[2].free_from = eta + 50  # the scheduled gate is still blocked by a late departure
    gates.append(Gate(id="R1", terminal="R", widebody=True, free_from=-999, free_until=10_000, remote=True))
    ramp_closed = weather_close if (brief and (brief.thunderstorm or brief.convective_nearby) and weather_close > -999) else -999

    tail_legs = flights[(flights.TAIL_NUMBER == row.TAIL_NUMBER) & (flights.MONTH == row.MONTH)
                        & (flights.DAY == row.DAY) & (flights.ORIGIN_AIRPORT == dest)]
    tail_legs = tail_legs[tail_legs.SD_MIN > sibt_min]
    if len(tail_legs):
        nl = tail_legs.sort_values("SD_MIN").iloc[0]
        next_leg = NextLeg(flight=f"{nl.AIRLINE}{int(nl.FLIGHT_NUMBER)}", destination=str(nl.DESTINATION_AIRPORT),
                           tobt=int(nl.SD_MIN) - sibt_min, source="DOT")
    else:
        next_leg = NextLeg(flight=f"{row.AIRLINE}{rng.randint(100, 2999)}", destination="(next leg)",
                           tobt=int(cal_turn[wake]) + rng.randint(5, 30), source="SIM")

    conns = flights[(flights.ORIGIN_AIRPORT == dest) & (flights.AIRLINE == row.AIRLINE) & (flights.MONTH == row.MONTH)
                    & (flights.DAY == row.DAY)]
    conns = conns[(conns.SD_MIN - sibt_min).between(max(eta, 0) + 45, max(eta, 0) + 180)].sort_values("SD_MIN")
    connections = [Connection(flight=f"{c.AIRLINE}{int(c.FLIGHT_NUMBER)}", destination=str(c.DESTINATION_AIRPORT),
                              pax=rng.randint(6, 45), departs_at=int(c.SD_MIN) - sibt_min)
                   for c in conns.head(3).itertuples()]

    # --- fuel --------------------------------------------------------------------
    alt, alt_nm = _nearest_alternate(dest)
    frf = FINAL_RESERVE_MIN * ac["hold_kg_min"]
    alt_fuel = int((alt_nm / ac["cruise_nm_min"] + 12) * ac["cruise_kg_min"])  # + missed approach & approach
    extra_min = rng.randint(14, 50)
    fob = frf + alt_fuel + extra_min * ac["hold_kg_min"]

    return Scenario(
        row_index=int(row.get("index", row.name)), flight=f"{row.AIRLINE}{int(row.FLIGHT_NUMBER)}",
        airline=str(row.AIRLINE), tail_number=str(row.TAIL_NUMBER), origin=str(row.ORIGIN_AIRPORT),
        destination=dest, destination_name=ref.name if ref else dest,
        date=f"{int(row.YEAR)}-{int(row.MONTH):02d}-{int(row.DAY):02d}",
        scheduled_arrival_hhmm=f"{int(row.SCHEDULED_ARRIVAL) % 2400:04d}", weather_delay_min=weather_delay,
        late_aircraft_delay_min=late, distance_mi=int(row.DISTANCE),
        weather_mode=weather_mode, weather=brief, weather_at_destination=dest_wx, weather_attribution=attribution,
        imc=imc, airport_icao=ref.icao if ref else dest,
        wake=wake, aircraft_type=ac["type"], eta_min=eta, fob_at_eta_kg=int(fob),
        hold_burn_kg_per_min=ac["hold_kg_min"], final_reserve_kg=int(frf), alternate=alt, alternate_fuel_kg=alt_fuel,
        connections=connections, runways=runways, taxiway_note=taxiway_note,
        gates=gates, scheduled_gate=gates[2].id, turnaround_min=turnaround, deice_required=deice,
        ramp_closed_until=ramp_closed, crew_shift_end_min=max(eta, 0) + rng.randint(55, 110), next_leg=next_leg,
    )


def scenario_id(row: pd.Series) -> str:
    return f"{row.AIRLINE}{int(row.FLIGHT_NUMBER)}-{int(row.YEAR)}{int(row.MONTH):02d}{int(row.DAY):02d}-{row.TAIL_NUMBER}"


def split_of(row: pd.Series) -> Literal["dev", "test"]:
    """Stable 80/20 dev/test assignment by a hash of the flight's identity."""
    return "test" if int(hashlib.sha256(scenario_id(row).encode()).hexdigest(), 16) % 5 == 0 else "dev"


_ = math  # (kept for callers that import math helpers from here)
