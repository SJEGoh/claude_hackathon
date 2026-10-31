"""The demo: one prompt sent to #ops-team that makes the Chief of Staff run the whole multi-agent flow.

DOT 2015 row 115187, UA1112 Washington Dulles -> Denver on 6 May 2015: WEATHER_DELAY 37, LATE_AIRCRAFT_DELAY 26.
That evening Denver had a thunderstorm with lightning (runways and ramp closed), a moderate-turbulence PIREP, the
scheduled stand was occupied and the fuel margin was tight, so every specialist has something to defend.
"""
DEMO_ROW = 115187
DEMO_PROMPT = ("We have a disruption. Load DOT 2015 row 115187: UA1112, Washington Dulles to Denver on 6 May 2015, "
               "WEATHER_DELAY 37 min and LATE_AIRCRAFT_DELAY 26 min, with that day's historical weather. Brief the "
               "team in three bullets, ask ATC for the earliest conflict-free landing slot and the Captain for the "
               "fuel situation, then run the negotiation, supervised, so ATC, the Captain and AOCC agree a revised "
               "arrival time, runway and stand.")
