# A-CDM Multi-Agent Disruption Negotiation (AUTONOMOUS SG 2026 · Challenge #2)

Three independent Claude specialists negotiate a revised landing time, runway and stand for a disrupted
US DOT 2015 flight, in at most 3 turns, chaired by a Chief of Staff. Plain code checks every plan, a
deterministic planner takes over when the agents don't converge, and a human can supervise at chosen
points.

| Agent | Role (A-CDM) | Private instruments |
|---|---|---|
| `agents/atc_agent.py` | Arrival Manager + ground movement | METAR, runway wind components, closures, wake minima, real + simulated traffic, AMAN free slots |
| `agents/pilot_agent.py` | Captain / FMS | FOB, FRF + ALTN, latest safe TLDT, PIREPs (turbulence/icing), crosswinds, real onward connections |
| `agents/ground_agent.py` | AOCC turnaround coordinator | stands, EIBT, next-leg TOBT (real DOT schedule), crew duty, ramp lightning closure, de-icing |
| `agents/moderator.py` | Chief of Staff (Duty Ops Manager) | public facts only; no vote |

## Talking to the team (Ops chat page, the default)
- **#ops-team group chat**: you, the Chief of Staff and the three specialists. Messages go to the Chief unless you
  `@ATC`, `@Pilot` or `@Ground` someone. The Chief uses tools to find and load DOT flights, ask specialists, run the
  bounded negotiation (which streams into the thread) and relay your decisions. You can answer in plain words or with
  the buttons.
- **Private DMs** with each agent. Every agent keeps its own memory per channel (`agents/conversation.py`), sees only
  its own instruments, and has deterministic tools: `check_plan` for its own domain, ATC `free_slots`, Pilot `fuel_at`.
  DMs are never shown to other agents.
- The roster sidebar shows each agent as idle, working, or waiting for you. Chats are appended to `logs/chat_*.jsonl`.

## Decision layer and emergencies
- **Costs** (`costs.py`): soft trade-offs are priced in USD with sourced, editable weights: fuel $/kg, aircraft time,
  missed connections, crew overtime, remote stand, and next-leg knock-on (delay $/min x the 0.76 propagation measured in
  the EDA). Costs only **rank plans that already pass every hard check**; safety is never weighed against money. Each
  result shows an itemised breakdown and a robustness score: the share of 300 random ±50% weight scenarios in which the
  plan stays cheapest. The Chief of Staff gets this table during negotiation and has a `compare_plans` tool in chat.
- **MAYDAY fast path** (`emergency.py`): a red button on both pages, or just type "MAYDAY…" to the Chief. The
  negotiation is abandoned and a deterministic plan is issued in under a millisecond: the earliest feasible landing
  with zero added holding (never "offset 0", which can be physically impossible), conflicting traffic re-sequenced,
  the right stand and services (remote stand with fire & rescue for technical failures, paramedics at the nearest gate
  for medical), and parallel checklists for ATC, Flight Crew and AOCC. If no runway is usable, the aircraft diverts.
  Agents only read back their actions; the plan never waits for them. Each emergency is logged under
  `logs/EMERGENCY_*.jsonl`.

## Real-world stress tests (`stress.py`, Control room → 🔥 Stress tests)
Each case replays a documented failure mode against the live engine and checks: no crash · never an unsafe agreed
plan · ≤ 3 turns · inside the time budget · log intact.
| Case | Real-world basis | What must happen |
|---|---|---|
| NATS_RECORD | UK NATS Aug 2023: two beacons with one name crashed primary and identical backup | bad/ambiguous records quarantined, rest of batch continues; fallback is independent code |
| LIGHTNING_RAMP | HKIA stops ramp work within 1 km; Changi suspends baggage handling | no contact stand before the ramp reopens |
| RADAR_OUTAGE | PHL TRACON/Newark 2025, ~90 s radar+radio loss | all agents down → independent planner + alarm |
| STAFFING_SHORTAGE | US ~3,500 controllers short (2025) | single runway, extra spacing, flaky agents, 20 s hard budget |
| RUNWAY_CLOSURE | disabled aircraft | closed runway never used |
| ROGUE_AGENTS | safety drill | unsafe ACCEPT overridden |
| STORM_FRONT_WAVE | arrivals compressed into remaining capacity | parallel negotiations on a stale snapshot; lock-protected ledger re-validates, re-plans, fuel priority; no wake conflict, no double-booked stand |

## Data
- `eda.py` produces `eda/EDA_REPORT.md` and `eda/calibration.json` (taxi-in, arrival rates, turnaround). Key findings: the file
  is pre-filtered, all rows already meet the rule, October is missing, 88% of rows are late-aircraft only, and inbound delay
  propagates to the next leg with correlation 0.76.
- `weather.py` handles weather in plain code with no LLM. It uses the IEM METAR and PIREP archives plus Open-Meteo ERA5 for
  the flight's real date, aviationweather.gov for live data, and OurAirports for runways. Every payload is cached with a
  SHA-256 and recorded in the consensus log.
- `weather_poller.py` refreshes the live cache on an interval.

## Safety and control
- Hard checks (`constraints.py`): wake separation (+1 min IMC), runway closure, crosswind ≤ 33 kt / tailwind ≤ 10 kt,
  FRF + ALTN, max 10 min holding in SEV PIREPs, stand size and availability, crew duty cap.
- Real-ID enums: agents can only name runways and stands that exist in the scenario. Plan numbers make stale replies
  get discarded.
- Autonomy tiers: AUTONOMOUS, SUPERVISED and STEP_BY_STEP. Decision points cover consensus review, turn review,
  deadlock, agent timeout or failure (wait and retry, keep the best safe plan, or fallback) and fallback review. Humans
  are only offered plans that pass every hard check.
- Alarms (`escalation.py`): on-screen banner, toast and sound, a webhook, a macOS notification and a tap-to-call duty
  manager link.
- Monitoring (`monitoring.py`): latency, tokens and USD for every call. Agents see a compact bus (a constraint ledger
  plus the latest turn) instead of the full transcript.
- Log: `logs/*.jsonl`, hash-chained and tamper-evident.

## Run
```bash
source .venv/bin/activate
python check_api.py                       # key in .env works?
streamlit run app.py                      # Ops chat + Control room pages
python run_negotiation.py --row 40        # CLI (add --autonomy SUPERVISED to be asked in the terminal)
python run_negotiation.py --row 40 --agents rules   # offline, free
python weather_poller.py --airports ORD ATL --interval 300
python batch_eval.py --agents rules --n 5           # free baseline
python batch_eval.py --split test --n 4             # Claude agents on held-out flights (~$0.2-0.5 per flight)
python -m unittest discover tests         # 22 offline tests
```
