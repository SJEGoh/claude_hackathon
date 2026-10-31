"""Deterministic aviation weather + airport reference data. No LLM touches any of this.

Sources (raw payloads are cached on disk with a SHA-256 so every scenario is reproducible and auditable):
  - IEM ASOS archive        historical METARs for the flight's real date      (mesonet.agron.iastate.edu)
  - IEM PIREP archive       historical pilot reports near the destination      (mesonet.agron.iastate.edu)
  - Open-Meteo archive      hourly ERA5 surface weather for the airport        (archive-api.open-meteo.com)
  - aviationweather.gov     LIVE METAR / TAF / PIREP (data API keeps ~15 days) (aviationweather.gov/api/data)
  - OurAirports             runway identifiers and true headings               (davidmegginson.github.io/ourairports-data)

aviationweather.gov cannot serve 2015, so historical scenarios use IEM (the archive AWC points users to) and
Open-Meteo; live mode uses aviationweather.gov. `weather_poller.py` refreshes the live cache on an interval.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel
from timezonefinder import TimezoneFinder

ROOT = Path(__file__).parent
CACHE = ROOT / "weather_cache"
REF = ROOT / "refdata"
HTTP_TIMEOUT_S = 20
USER_AGENT = "autonomous-sg-2026-atm-agents/1.0 (hackathon research)"

OURAIRPORTS = {
    "airports": "https://davidmegginson.github.io/ourairports-data/airports.csv",
    "runways": "https://davidmegginson.github.io/ourairports-data/runways.csv",
}
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_PIREPS = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/pireps.py"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
AWC = "https://aviationweather.gov/api/data"

# Typical narrow/wide-body operator limits, used as hard checks (configurable).
MAX_CROSSWIND_KT = 33
MAX_TAILWIND_KT = 10
MIN_RUNWAY_LENGTH_FT = 7000
PIREP_RADIUS_NM = 100
PIREP_WINDOW_H = 3
PIREP_MAX_ALT_FT = 18000  # holding / approach altitudes
INTENSITY = ["NONE", "LGT", "MOD", "SEV"]

Intensity = Literal["NONE", "LGT", "MOD", "SEV"]
FlightCategory = Literal["VFR", "MVFR", "IFR", "LIFR", "UNKNOWN"]


# ============================================================================
# Cached HTTP
# ============================================================================

class SourceRecord(BaseModel):
    source: str
    url: str
    fetched_at: str
    sha256: str
    cached: bool
    ok: bool
    note: str = ""


def _fetch(source: str, url: str, params: dict, cache_key: str, max_age_s: Optional[float] = None
           ) -> tuple[Optional[str], SourceRecord]:
    """GET with an on-disk cache. Historical data never expires; live data passes max_age_s."""
    path = CACHE / source / f"{cache_key}.txt"
    meta = path.with_suffix(".json")
    full_url = requests.Request("GET", url, params=params).prepare().url or url
    if path.exists() and meta.exists():
        m = json.loads(meta.read_text())
        if max_age_s is None or time.time() - m["epoch"] < max_age_s:
            body = path.read_text()
            return body, SourceRecord(source=source, url=full_url, fetched_at=m["fetched_at"],
                                      sha256=m["sha256"], cached=True, ok=True)
    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        body = r.text
    except requests.RequestException as e:
        if path.exists():  # stale cache beats nothing, and we say so
            m = json.loads(meta.read_text())
            return path.read_text(), SourceRecord(source=source, url=full_url, fetched_at=m["fetched_at"],
                                                  sha256=m["sha256"], cached=True, ok=True,
                                                  note=f"refresh failed ({type(e).__name__}); using stale cache")
        return None, SourceRecord(source=source, url=full_url, fetched_at="", sha256="", cached=False, ok=False,
                                  note=f"{type(e).__name__}: {e}")
    digest = hashlib.sha256(body.encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    meta.write_text(json.dumps({"fetched_at": now, "epoch": time.time(), "sha256": digest, "url": full_url}))
    return body, SourceRecord(source=source, url=full_url, fetched_at=now, sha256=digest, cached=False, ok=True)


# ============================================================================
# Airport reference data (OurAirports)
# ============================================================================

class RunwayEnd(BaseModel):
    ident: str
    heading_true: float
    length_ft: int
    physical: str  # e.g. "10L/28R"


class AirportRef(BaseModel):
    iata: str
    icao: str
    name: str
    lat: float
    lon: float
    tz: str
    runway_ends: list[RunwayEnd]


def _ourairports(kind: str) -> list[dict]:
    path = REF / f"ourairports_{kind}.csv"
    if not path.exists():
        REF.mkdir(exist_ok=True)
        r = requests.get(OURAIRPORTS[kind], timeout=60, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        path.write_text(r.text)
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


_AIRPORT_CACHE: dict[str, AirportRef] = {}
_TF = TimezoneFinder()


def airport_ref(iata: str) -> Optional[AirportRef]:
    if iata in _AIRPORT_CACHE:
        return _AIRPORT_CACHE[iata]
    rows = [a for a in _ourairports("airports") if a["iata_code"] == iata]
    if not rows:
        return None
    a = max(rows, key=lambda r: r["type"] == "large_airport")
    ends = []
    for rw in _ourairports("runways"):
        if rw["airport_ident"] != a["ident"] or rw["closed"] == "1" or not rw["length_ft"]:
            continue
        length = int(float(rw["length_ft"]))
        if length < MIN_RUNWAY_LENGTH_FT:
            continue
        for side, other in (("le", "he"), ("he", "le")):
            ident, hdg = rw[f"{side}_ident"], rw[f"{side}_heading_degT"]
            if not ident or ident.startswith("H"):
                continue
            if not hdg:  # derive from the runway number when the true heading is missing
                num = re.match(r"\d+", ident)
                if not num:
                    continue
                hdg = int(num.group()) * 10
            ends.append(RunwayEnd(ident=ident, heading_true=float(hdg), length_ft=length,
                                  physical=f"{rw['le_ident']}/{rw['he_ident']}"))
    lat, lon = float(a["latitude_deg"]), float(a["longitude_deg"])
    ref = AirportRef(iata=iata, icao=a["ident"], name=a["name"], lat=lat, lon=lon,
                     tz=_TF.timezone_at(lat=lat, lng=lon) or "UTC", runway_ends=ends)
    _AIRPORT_CACHE[iata] = ref
    return ref


# ============================================================================
# Weather brief (the deterministic product every agent and validator reads)
# ============================================================================

class Pirep(BaseModel):
    time_utc: str
    aircraft: str
    altitude_ft: Optional[int]
    turbulence: Intensity
    icing: Intensity
    raw: str
    distance_nm: float


class RunwayWind(BaseModel):
    runway: str
    headwind_kt: float
    crosswind_kt: float  # includes gust
    tailwind_kt: float
    within_limits: bool


class WeatherBrief(BaseModel):
    mode: Literal["HISTORICAL", "LIVE"]
    airport_icao: str
    valid_local: str
    metar: Optional[str]
    taf: Optional[str] = None
    wind_dir_deg: Optional[float]
    wind_kt: Optional[float]
    gust_kt: Optional[float]
    visibility_sm: Optional[float]
    ceiling_ft: Optional[int]
    flight_category: FlightCategory
    thunderstorm: bool
    convective_nearby: bool  # CB/TCU in the METAR or a thunderstorm weather code from Open-Meteo
    freezing_precip_or_snow: bool
    temperature_c: Optional[float]
    pireps: list[Pirep]
    hold_turbulence: Intensity
    hold_icing: Intensity
    runway_winds: list[RunwayWind]
    open_meteo: dict
    sources: list[SourceRecord]

    def summary(self) -> str:
        wind = (f"{self.wind_dir_deg:03.0f}/{self.wind_kt:.0f}kt" + (f" G{self.gust_kt:.0f}" if self.gust_kt else "")
                if self.wind_kt is not None and self.wind_dir_deg is not None else "wind n/a")
        return (f"{self.airport_icao} {self.flight_category} {wind}, vis {self.visibility_sm or 'n/a'} sm, ceiling "
                f"{self.ceiling_ft or 'none'} ft{', TS' if self.thunderstorm else ''}"
                f"{', CB/convection nearby' if self.convective_nearby and not self.thunderstorm else ''}"
                f"{', freezing precip/snow' if self.freezing_precip_or_snow else ''}; holding-altitude PIREPs: "
                f"turbulence {self.hold_turbulence}, icing {self.hold_icing} ({len(self.pireps)} reports)")


def _flight_category(vis: Optional[float], ceiling: Optional[int]) -> FlightCategory:
    if vis is None and ceiling is None:
        return "UNKNOWN"
    v = vis if vis is not None else 10
    c = ceiling if ceiling is not None else 99999
    if v < 1 or c < 500:
        return "LIFR"
    if v < 3 or c < 1000:
        return "IFR"
    if v <= 5 or c <= 3000:
        return "MVFR"
    return "VFR"


def _ceiling(metar: str) -> Optional[int]:
    layers = [int(h) * 100 for cov, h in re.findall(r"\b(BKN|OVC|VV)(\d{3})", metar)]
    return min(layers) if layers else None


def _visibility(metar: str) -> Optional[float]:
    m = re.search(r"\s(M?)(\d+)?\s?(\d/\d)?SM\b", metar)
    if not m:
        return None
    whole = float(m.group(2) or 0)
    frac = eval(m.group(3)) if m.group(3) else 0.0  # noqa: S307 - "1/2" style fraction from a regex match
    return whole + frac


def _intensity(text: str) -> Intensity:
    t = (text or "").upper()
    if "SEV" in t or "EXTRM" in t:
        return "SEV"
    if "MOD" in t:
        return "MOD"
    if "LGT" in t or "LIGHT" in t:
        return "LGT"
    return "NONE"


def _distance_nm(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 3440.065 * 2 * math.asin(math.sqrt(h))


def runway_winds(ends: list[RunwayEnd], wdir: Optional[float], wspd: Optional[float], gust: Optional[float]
                 ) -> list[RunwayWind]:
    out = []
    for e in ends:
        if wdir is None or wspd is None:
            out.append(RunwayWind(runway=e.ident, headwind_kt=0, crosswind_kt=0, tailwind_kt=0, within_limits=True))
            continue
        angle = math.radians(wdir - e.heading_true)
        head = wspd * math.cos(angle)
        cross = abs((gust or wspd) * math.sin(angle))
        tail = max(0.0, -head)
        out.append(RunwayWind(runway=e.ident, headwind_kt=round(head, 1), crosswind_kt=round(cross, 1),
                              tailwind_kt=round(tail, 1),
                              within_limits=cross <= MAX_CROSSWIND_KT and tail <= MAX_TAILWIND_KT))
    return out


def _station_for_iem(ref: AirportRef) -> str:
    return ref.icao[1:] if ref.icao.startswith("K") and len(ref.icao) == 4 else ref.icao


def historical_brief(ref: AirportRef, local_dt: datetime) -> WeatherBrief:
    """Weather at `local_dt` (airport local time) from the IEM archives and Open-Meteo."""
    tz = ZoneInfo(ref.tz)
    utc_dt = local_dt.replace(tzinfo=tz).astimezone(timezone.utc)
    day = utc_dt.date()
    sources: list[SourceRecord] = []

    # --- METAR (IEM ASOS archive) -------------------------------------------
    start, end = utc_dt - timedelta(hours=3), utc_dt + timedelta(hours=3)
    params = {"station": _station_for_iem(ref), "data": ["metar", "sknt", "drct", "gust", "vsby", "tmpf"],
              "year1": start.year, "month1": start.month, "day1": start.day,
              "year2": end.year, "month2": end.month, "day2": end.day + 0, "tz": "Etc/UTC",
              "format": "onlycomma", "missing": "empty", "report_type": [3, 4]}
    end_plus = end + timedelta(days=1)
    params.update(year2=end_plus.year, month2=end_plus.month, day2=end_plus.day)
    body, rec = _fetch("iem_asos", IEM_ASOS, params, f"{ref.icao}_{day.isoformat()}")
    sources.append(rec)
    obs = []
    if body:
        for row in csv.DictReader(io.StringIO(body)):
            try:
                t = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                continue
            obs.append((abs((t - utc_dt).total_seconds()), t, row))
    metar = wdir = wspd = gust = vis = temp_c = None
    if obs:
        _, _, row = min(obs, key=lambda o: o[0])
        metar = row.get("metar") or None
        num = lambda k: float(row[k]) if row.get(k) not in (None, "", "M") else None  # noqa: E731
        wdir, wspd, gust, vis = num("drct"), num("sknt"), num("gust"), num("vsby")
        if num("tmpf") is not None:
            temp_c = round((num("tmpf") - 32) * 5 / 9, 1)

    # --- Open-Meteo archive ---------------------------------------------------
    om_params = {"latitude": ref.lat, "longitude": ref.lon, "start_date": day.isoformat(), "end_date": day.isoformat(),
                 "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation,snowfall,"
                           "temperature_2m,weather_code,cloud_cover_low",
                 "wind_speed_unit": "kn", "timezone": "UTC"}
    body, rec = _fetch("open_meteo", OPEN_METEO_ARCHIVE, om_params, f"{ref.icao}_{day.isoformat()}")
    sources.append(rec)
    om = {}
    if body:
        hourly = json.loads(body).get("hourly", {})
        times = hourly.get("time", [])
        key = utc_dt.strftime("%Y-%m-%dT%H:00")
        if key in times:
            i = times.index(key)
            om = {k: v[i] for k, v in hourly.items() if k != "time"}
    # Open-Meteo fills gaps when the METAR is missing (and says so in the brief).
    if wspd is None and om:
        wdir, wspd, gust = om.get("wind_direction_10m"), om.get("wind_speed_10m"), om.get("wind_gusts_10m")
    if temp_c is None and om:
        temp_c = om.get("temperature_2m")

    # --- PIREPs (IEM archive) ---------------------------------------------------
    # The 2015 archive is sparse, so pull the nationwide feed in 6-hour UTC blocks (cached and shared by
    # every scenario in that block) and filter by distance and altitude here.
    ps, pe = utc_dt - timedelta(hours=PIREP_WINDOW_H), utc_dt + timedelta(hours=PIREP_WINDOW_H)
    block = ps.replace(hour=ps.hour // 6 * 6, minute=0, second=0, microsecond=0)
    pireps, seen = [], set()
    while block < pe:
        b_end = block + timedelta(hours=6)
        p_params = {"sts": block.strftime("%Y-%m-%dT%H:%MZ"), "ets": b_end.strftime("%Y-%m-%dT%H:%MZ"),
                    "fmt": "csv", "artcc": "_ALL"}
        body, rec = _fetch("iem_pireps", IEM_PIREPS, p_params, f"ALL_{block.strftime('%Y%m%dT%H')}")
        sources.append(rec)
        for row in csv.DictReader(io.StringIO(body or "")):
            report = row.get("REPORT", "")
            if report in seen:
                continue
            seen.add(report)
            try:
                t = datetime.strptime(row["VALID"], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                alt = int(float(row["FL"])) if row.get("FL") not in (None, "", "None") else None
                d = _distance_nm(ref.lat, ref.lon, float(row["LAT"]), float(row["LON"]))
            except (KeyError, ValueError):
                continue
            if not ps <= t <= pe or d > PIREP_RADIUS_NM or (alt is not None and alt > PIREP_MAX_ALT_FT):
                continue
            pireps.append(Pirep(time_utc=row["VALID"], aircraft=row.get("AIRCRAFT", ""), altitude_ft=alt,
                                turbulence=_intensity(row.get("TURBULENCE", "")),
                                icing=_intensity(row.get("ICING", "")), raw=report, distance_nm=round(d, 1)))
        block = b_end
    return _assemble("HISTORICAL", ref, local_dt, metar, None, wdir, wspd, gust, vis, temp_c, pireps, om, sources)


def live_brief(ref: AirportRef, max_age_s: float = 600) -> WeatherBrief:
    """Current weather from the aviationweather.gov data API (cached for max_age_s)."""
    sources: list[SourceRecord] = []
    body, rec = _fetch("awc_metar", f"{AWC}/metar", {"ids": ref.icao, "format": "json"}, ref.icao, max_age_s)
    sources.append(rec)
    metar = wdir = wspd = gust = vis = temp_c = None
    if body:
        data = json.loads(body) or [{}]
        m = data[0]
        metar = m.get("rawOb")
        wdir = m.get("wdir") if isinstance(m.get("wdir"), (int, float)) else None  # "VRB" -> None
        wspd, gust, temp_c = m.get("wspd"), m.get("wgst"), m.get("temp")
        v = m.get("visib")
        vis = 10.0 if v == "10+" else (float(v) if isinstance(v, (int, float)) else None)
    body, rec = _fetch("awc_taf", f"{AWC}/taf", {"ids": ref.icao, "format": "json"}, ref.icao, max_age_s)
    sources.append(rec)
    taf = (json.loads(body) or [{}])[0].get("rawTAF") if body else None
    body, rec = _fetch("awc_pirep", f"{AWC}/pirep",
                       {"id": ref.icao, "distance": PIREP_RADIUS_NM, "age": 2, "format": "json"}, ref.icao, max_age_s)
    sources.append(rec)
    pireps = []
    for p in (json.loads(body) if body else []) or []:
        alt = (p.get("fltLvl") or 0) * 100 or None
        if alt and alt > PIREP_MAX_ALT_FT:
            continue
        pireps.append(Pirep(time_utc=str(p.get("obsTime")), aircraft=p.get("acType") or "", altitude_ft=alt,
                            turbulence=_intensity(p.get("tbInt1", "")), icing=_intensity(p.get("icgInt1", "")),
                            raw=p.get("rawOb", ""),
                            distance_nm=round(_distance_nm(ref.lat, ref.lon, p.get("lat", ref.lat),
                                                           p.get("lon", ref.lon)), 1)))
    now_local = datetime.now(ZoneInfo(ref.tz)).replace(tzinfo=None)
    return _assemble("LIVE", ref, now_local, metar, taf, wdir, wspd, gust, vis, temp_c, pireps, {}, sources)


def _assemble(mode, ref, local_dt, metar, taf, wdir, wspd, gust, vis, temp_c, pireps, om, sources) -> WeatherBrief:
    if metar and vis is None:
        vis = _visibility(metar)
    ceiling = _ceiling(metar) if metar else None
    wx = metar or ""
    om_code = om.get("weather_code") or 0
    thunder = bool(re.search(r"\b(\+|-|VC)?TS", wx)) or 95 <= om_code <= 99
    convective = thunder or bool(re.search(r"\b(CB|TCU)\b|\d{3}(CB|TCU)\b|CBMAM", wx))
    frozen = bool(re.search(r"\b(\+|-)?(FZRA|FZDZ|SN|PL|GS)\b", wx)) or (om.get("snowfall") or 0) > 0 or (
        temp_c is not None and temp_c <= 1 and (om.get("precipitation") or 0) > 0)
    worst = lambda attr: max((getattr(p, attr) for p in pireps), key=INTENSITY.index, default="NONE")  # noqa: E731
    return WeatherBrief(
        mode=mode, airport_icao=ref.icao, valid_local=local_dt.strftime("%Y-%m-%d %H:%M"), metar=metar, taf=taf,
        wind_dir_deg=wdir, wind_kt=wspd, gust_kt=gust, visibility_sm=vis, ceiling_ft=ceiling,
        flight_category=_flight_category(vis, ceiling), thunderstorm=thunder, convective_nearby=convective, freezing_precip_or_snow=frozen,
        temperature_c=temp_c, pireps=pireps, hold_turbulence=worst("turbulence"), hold_icing=worst("icing"),
        runway_winds=runway_winds(ref.runway_ends, wdir, wspd, gust), open_meteo=om, sources=sources,
    )
