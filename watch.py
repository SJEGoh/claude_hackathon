"""Automated weather watch: deterministic checks on a watchlist of airports, notifying ONLY on change.

Every interval, for each airport it reads live METAR, TAF and PIREPs (aviationweather.gov), the next hours of
Open-Meteo forecast (precipitation, CAPE, storm codes, gusts, visibility) and convective SIGMETs, and runs:
  thunderstorm   METAR TS/VCTS/CB, TAF TS in the next hours, forecast storm codes, high CAPE
  rain           heavy rain now or forecast; suggested runway condition code
  wind           crosswind/tailwind per runway against the (runway-condition-adjusted) limits; gusts
  visibility     IFR, and LIFR = low-visibility procedures (CAT II/III) likely
  windshear      wind shear in the METAR, TAF or PIREPs
  turbulence_icing  MOD/SEV reports at approach and holding altitudes
  freezing       freezing rain/drizzle, snow, ice pellets: de-icing and contaminated runway
  sigmet         convective SIGMET within the configured radius
  data_health    a feed failed or is stale: decisions would rest on old data (the NATS lesson)

Alerts are de-duplicated: you hear about a condition when it appears, gets worse, or clears, not every poll.
Run in the background:  python watch.py            (loop)      python watch.py --once
The app also runs the same check on a timer from the sidebar.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

import ops_settings
import weather as W
from escalation import push

STATE = W.CACHE / "watch_state.json"
EVENTS = W.CACHE / "watch_events.jsonl"
SEV_RANK = {"INFO": 0, "WARNING": 1, "ALARM": 2}
Severity = Literal["INFO", "WARNING", "ALARM"]


class Alert(BaseModel):
    airport: str
    check: str
    severity: Severity
    title: str
    detail: str
    observed: str

    @property
    def key(self) -> str:
        return f"{self.airport}:{self.check}:{self.title}"


class WatchEvent(BaseModel):
    at: str
    kind: Literal["NEW", "ESCALATED", "CLEARED"]
    alert: Alert
    channels: list[str] = []


# ============================================================================
# Feeds
# ============================================================================

def _json(source: str, url: str, params: dict, key: str, max_age_s: float):
    body, rec = W._fetch(source, url, params, key, max_age_s)
    try:
        return (json.loads(body) if body else None), rec
    except json.JSONDecodeError:
        return None, rec


def _forecast(ref: W.AirportRef, max_age_s: float) -> tuple[dict, W.SourceRecord]:
    data, rec = _json("om_forecast", "https://api.open-meteo.com/v1/forecast",
                      {"latitude": ref.lat, "longitude": ref.lon, "forecast_hours": 4, "wind_speed_unit": "kn",
                       "hourly": "precipitation,cape,weather_code,wind_gusts_10m,visibility"}, ref.icao, max_age_s)
    return (data or {}).get("hourly", {}), rec


def _sigmets(max_age_s: float) -> tuple[list, list[W.SourceRecord]]:
    out, recs = [], []
    for kind, url, params in (("awc_airsigmet", f"{W.AWC}/airsigmet", {"format": "json", "hazard": "conv"}),
                              ("awc_isigmet", f"{W.AWC}/isigmet", {"format": "json", "hazard": "ts"})):
        data, rec = _json(kind, url, params, "all", max_age_s)
        recs.append(rec)
        out += data or []
    return out, recs


def _near_sigmet(ref: W.AirportRef, sigmets: list, radius_nm: float) -> Optional[str]:
    for sg in sigmets:
        coords = sg.get("coords") or []
        pts = [(c.get("lat"), c.get("lon")) for c in coords if isinstance(c, dict) and c.get("lat") is not None]
        if not pts:
            continue
        inside = _point_in_polygon(ref.lat, ref.lon, pts)
        nearest = min(W._distance_nm(ref.lat, ref.lon, la, lo) for la, lo in pts)
        if inside or nearest <= radius_nm:
            raw = (sg.get("rawAirSigmet") or sg.get("rawSigmet") or "").replace("\n", " ")
            return f"{'inside' if inside else f'{nearest:.0f} NM from'} convective SIGMET: {raw[:160]}"
    return None


def _point_in_polygon(lat: float, lon: float, poly: list[tuple[float, float]]) -> bool:
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


# ============================================================================
# Checks
# ============================================================================

def run_checks(iata: str) -> list[Alert]:
    cfg = ops_settings.get()
    ref = W.airport_ref(iata.upper())
    now = datetime.now(timezone.utc).strftime("%H:%MZ")
    if ref is None:
        return [Alert(airport=iata, check="data_health", severity="WARNING", title="Unknown airport",
                      detail=f"{iata} not found in OurAirports", observed=now)]
    age = max(60, cfg.watch_interval_s - 30)
    brief = W.live_brief(ref, max_age_s=age)
    fc, fc_rec = _forecast(ref, age)
    sigmets, sg_recs = _sigmets(max(age, 600)) if cfg.checks.get("sigmet") else ([], [])
    taf_data, _ = _json("awc_taf", f"{W.AWC}/taf", {"ids": ref.icao, "format": "json"}, ref.icao, age)
    taf = (taf_data or [{}])[0] if taf_data else {}
    metar = brief.metar or ""
    A: list[Alert] = []

    def add(check, sev, title, detail):
        if cfg.checks.get(check, True):
            A.append(Alert(airport=ref.icao, check=check, severity=sev, title=title, detail=detail, observed=now))

    # thunderstorm / lightning ----------------------------------------------------------------
    horizon = datetime.now(timezone.utc) + timedelta(hours=3)
    taf_ts = [f for f in taf.get("fcsts", []) if "TS" in (f.get("wxString") or "")
              and datetime.fromtimestamp(f.get("timeFrom", 0), timezone.utc) <= horizon]
    codes = fc.get("weather_code", [])[:3]
    if re.search(r"(?<![A-Z])(\+|-)?TS", metar.split(" RMK")[0]):
        add("thunderstorm", "ALARM", "Thunderstorm at the field",
            f"METAR {metar}. Expect lightning ramp holds: handling stops, stands may block.")
    elif brief.convective_nearby:
        add("thunderstorm", "WARNING", "Convection in the vicinity (CB/TCU/VCTS)", f"METAR {metar}")
    if taf_ts:
        f = taf_ts[0]
        add("thunderstorm", "WARNING", "Thunderstorm forecast within 3 h",
            f"TAF {f.get('fcstChange') or 'FM'} {datetime.fromtimestamp(f['timeFrom'], timezone.utc):%H%MZ}-"
            f"{datetime.fromtimestamp(f['timeTo'], timezone.utc):%H%MZ}: {f.get('wxString')}")
    if any(95 <= (c or 0) <= 99 for c in codes):
        add("thunderstorm", "WARNING", "Model forecasts thunderstorms (next 3 h)", f"Open-Meteo weather codes {codes}")
    if max(fc.get("cape", [0])[:3] or [0]) >= 2500:
        add("thunderstorm", "INFO", "High convective potential", f"CAPE up to {max(fc['cape'][:3]):.0f} J/kg")

    # rain ----------------------------------------------------------------------------------------------
    heavy_now = bool(re.search(r"\+(SH|TS)?RA", metar))
    rain_now = bool(re.search(r"(?<![A-Z])[-+]?(SH|TS)?RA", metar.split(" RMK")[0]))
    peak = max(fc.get("precipitation", [0])[:3] or [0])
    if heavy_now or peak >= cfg.heavy_rain_mm_h:
        add("rain", "WARNING", "Heavy rain", f"{'+RA in METAR; ' if heavy_now else ''}forecast peak {peak:.1f} mm/h. "
            "Suggest RWYCC 3 or lower if standing water is reported; expect longer runway occupancy.")
    elif rain_now:
        add("rain", "INFO", "Rain at the field", "Wet runway: suggest RWYCC 5 in Ops settings.")

    # wind -------------------------------------------------------------------------------------------
    xw_lim = cfg.crosswind_limit()
    winds = brief.runway_winds
    if winds:
        out = [w for w in winds if w.crosswind_kt > xw_lim or w.tailwind_kt > cfg.max_tailwind_kt]
        if len(out) == len(winds):
            add("wind", "ALARM", "All runways out of wind limits", f"limits {xw_lim:.0f} kt cross / "
                f"{cfg.max_tailwind_kt:.0f} kt tail; {', '.join(f'{w.runway} x{w.crosswind_kt}/t{w.tailwind_kt}' for w in winds)}")
        elif out:
            add("wind", "WARNING", "Runways out of wind limits", ", ".join(f"{w.runway} cross {w.crosswind_kt} tail "
                                                                         f"{w.tailwind_kt} kt" for w in out))
    if (brief.gust_kt or 0) >= cfg.gust_alert_kt:
        add("wind", "WARNING", f"Gusts {brief.gust_kt:.0f} kt", f"METAR {metar}")

    # visibility ---------------------------------------------------------------------------------------
    if brief.flight_category == "LIFR":
        add("visibility", "ALARM", "LIFR: low-visibility procedures likely",
            f"vis {brief.visibility_sm} sm, ceiling {brief.ceiling_ft} ft. Consider approach mode ILS CAT II/III.")
    elif brief.flight_category == "IFR":
        add("visibility", "WARNING", "IFR conditions", f"vis {brief.visibility_sm} sm, ceiling {brief.ceiling_ft} ft")

    # windshear ---------------------------------------------------------------------------------------
    taf_ws = [f for f in taf.get("fcsts", []) if f.get("wshearSpd")]
    if re.search(r"\bWS\b|WIND ?SHEAR", metar) or taf_ws or any("LLWS" in p.raw or "WS" in p.raw.split() for p in brief.pireps):
        add("windshear", "ALARM", "Wind shear reported or forecast",
            "METAR/TAF/PIREP wind shear: expect go-arounds; consider increased spacing.")

    # turbulence / icing ------------------------------------------------------------------------------
    for attr, label in (("hold_turbulence", "turbulence"), ("hold_icing", "icing")):
        level = getattr(brief, attr)
        if level == "SEV":
            add("turbulence_icing", "ALARM", f"Severe {label} reported (≤ FL180)", "Holding there must be minimised.")
        elif level == "MOD":
            add("turbulence_icing", "WARNING", f"Moderate {label} reported (≤ FL180)", "PIREPs near the field.")

    # freezing -------------------------------------------------------------------------------------------------
    if brief.freezing_precip_or_snow or re.search(r"FZRA|FZDZ|(?<![A-Z])[-+]?SN|PL", metar.split(" RMK")[0]):
        add("freezing", "WARNING", "Freezing precipitation / snow",
            "De-icing needed (longer turnarounds); runway contamination: set RWYCC 3 or lower.")

    # SIGMET ---------------------------------------------------------------------------------------------------
    near = _near_sigmet(ref, sigmets, cfg.sigmet_radius_nm) if cfg.checks.get("sigmet") else None
    if near:
        add("sigmet", "WARNING", "Convective SIGMET near the field", near)

    # data health -------------------------------------------------------------------------------------------
    bad = [r for r in [*brief.sources, fc_rec, *sg_recs] if not r.ok or r.note]
    if bad:
        add("data_health", "WARNING", "Weather feed degraded", "; ".join(f"{r.source}: {r.note or 'unavailable'}" for r in bad))
    return A


# ============================================================================
# Change detection + notification
# ============================================================================

def poll(airports: Optional[list[str]] = None) -> list[WatchEvent]:
    """Run every check, compare with the last known state, notify on NEW / ESCALATED / CLEARED."""
    cfg = ops_settings.get()
    state: dict = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    current: dict[str, Alert] = {}
    for iata in airports or cfg.watch_airports:
        try:
            for a in run_checks(iata):
                current[a.key] = a
        except Exception as e:  # noqa: BLE001 - one airport's failure must not stop the watch
            a = Alert(airport=iata, check="data_health", severity="WARNING", title="Check failed",
                      detail=f"{type(e).__name__}: {e}", observed=datetime.now(timezone.utc).strftime("%H:%MZ"))
            current[a.key] = a
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    events = []
    for key, a in current.items():
        old = state.get(key)
        if old is None:
            events.append(WatchEvent(at=now, kind="NEW", alert=a))
        elif SEV_RANK[a.severity] > SEV_RANK[old["severity"]]:
            events.append(WatchEvent(at=now, kind="ESCALATED", alert=a))
    for key, old in state.items():
        if key not in current:
            events.append(WatchEvent(at=now, kind="CLEARED", alert=Alert(**old)))
    for ev in events:
        if ev.kind != "CLEARED" and SEV_RANK[ev.alert.severity] >= SEV_RANK[cfg.push_min_severity] or (
                ev.kind == "CLEARED" and SEV_RANK[ev.alert.severity] >= SEV_RANK[cfg.push_min_severity]):
            prefix = "✅ CLEARED" if ev.kind == "CLEARED" else f"{ev.alert.severity}"
            ev.channels = push(f"{prefix} {ev.alert.airport}: {ev.alert.title}", ev.alert.detail, ev.alert.severity
                               if ev.kind != "CLEARED" else "INFO")
    W.CACHE.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({k: a.model_dump() for k, a in current.items()}, indent=1), encoding="utf-8")
    with EVENTS.open("a", encoding="utf-8") as f:
        for ev in events:
            f.write(ev.model_dump_json() + "\n")
    return events


def active_alerts() -> list[Alert]:
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    return sorted((Alert(**a) for a in state.values()), key=lambda a: (-SEV_RANK[a.severity], a.airport))


def recent_events(n: int = 50) -> list[WatchEvent]:
    if not EVENTS.exists():
        return []
    return [WatchEvent(**json.loads(line)) for line in EVENTS.read_text(encoding="utf-8").splitlines()[-n:]][::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        for ev in poll():
            print(f"{ev.at} {ev.kind:<9} {ev.alert.severity:<7} {ev.alert.airport} {ev.alert.title}: {ev.alert.detail[:100]}"
                  + (f"  -> {', '.join(ev.channels)}" if ev.channels else ""))
        if args.once:
            break
        time.sleep(max(60, ops_settings.get().watch_interval_s))


_ = math, Path
if __name__ == "__main__":
    main()
