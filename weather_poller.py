"""Refresh the LIVE weather cache from aviationweather.gov on a fixed interval.

The agents never fetch weather. This process writes it to weather_cache/, the Streamlit app and the
scenario builder read it from there, and every payload carries its fetch time and SHA-256.

Run:  python weather_poller.py --airports ORD ATL DFW --interval 300
      python weather_poller.py --once
"""
import argparse
import json
import time
from datetime import datetime, timezone

import weather as W

LIVE_INDEX = W.CACHE / "live_index.json"


def poll(airports: list[str]) -> dict:
    index = json.loads(LIVE_INDEX.read_text(encoding="utf-8")) if LIVE_INDEX.exists() else {}
    for iata in airports:
        ref = W.airport_ref(iata)
        if ref is None:
            print(f"{iata}: not in OurAirports, skipped")
            continue
        brief = W.live_brief(ref, max_age_s=0)  # force refresh
        index[iata] = {"polled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "summary": brief.summary(), "metar": brief.metar,
                       "sources_ok": all(s.ok for s in brief.sources),
                       "notes": [s.note for s in brief.sources if s.note]}
        print(f"{index[iata]['polled_at']} {brief.summary()}")
    W.CACHE.mkdir(exist_ok=True)
    LIVE_INDEX.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--airports", nargs="+", default=["ORD", "ATL", "DFW", "DEN", "LAX", "SFO", "IAH", "LAS"])
    ap.add_argument("--interval", type=int, default=300, help="seconds between polls (AWC asks for >= 60)")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    interval = max(60, args.interval)
    while True:
        poll(args.airports)
        if args.once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
