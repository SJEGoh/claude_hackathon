# A-CDM Multi-Agent Disruption Negotiation (AUTONOMOUS SG 2026 · Challenge #2)

Three independent Claude specialists negotiate a revised landing time, runway and stand for a disrupted
US DOT 2015 flight, in at most 3 turns, chaired by a Chief of Staff. The controller, pilot and ground lead then
ratify the agreed plan. Plain code checks every plan, and a deterministic planner takes over when the agents don't
converge or the stakeholders reject the plan twice.

## From consensus to a committed plan
1. **Debate:** up to 3 turns. No consensus means the deterministic planner decides.
2. **Summary:** the Chief of Staff sends the stakeholders the plan, every specialist's reasoning and its own. Safety
   figures come from code; Claude only writes the narrative, and the brief still goes out if that call fails.
3. **Vote:** each stakeholder agrees or disagrees with a reason. All three agree: the plan is committed.
4. **New round:** any disagreement rejects the plan. The Chief collates the objections, which go to every specialist,
   and a new debate of up to 3 turns starts. The rejected plan can't be kept as the "best safe plan so far".
5. **Second vote:** if that plan is rejected too, the deterministic planner decides. Its plan can match one the
   stakeholders rejected, because it can't read their reasons; the fallback reason says so when that happens.

People vote in SUPERVISED and STEP_BY_STEP. In AUTONOMOUS, and for anyone left out of a vote, the AI agent answers, and
one that can't answer counts as disagreeing. The log and the outcome record who gave each vote. The vote card is one
shared screen with no sign-in, so the log can't tell which person pressed which button.

| Agent | Role (A-CDM) | Private instruments |
|---|---|---|
| `agents/atc_agent.py` | Arrival Manager + ground movement | METAR, runway wind components, closures, wake minima, runway capacity, real + simulated traffic, AMAN free slots |
| `agents/pilot_agent.py` | Captain / FMS | FOB, FRF + ALTN + contingency, latest safe TLDT, flight crew duty, PIREPs (turbulence/icing), crosswinds, real onward connections |
| `agents/ground_agent.py` | AOCC turnaround coordinator | stands, spare stands, gate-change notice, EIBT, next-leg TOBT (real DOT schedule), crew shifts and overtime, ramp lightning closure, de-icing |
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
plan · ≤ 3 turns per round · inside the time budget · log intact.
| Case | Real-world basis | What must happen |
|---|---|---|
| NATS_RECORD | UK NATS Aug 2023: two beacons with one name crashed primary and identical backup | bad/ambiguous records quarantined, rest of batch continues; fallback is independent code |
| LIGHTNING_RAMP | HKIA stops ramp work within 1 km; Changi suspends baggage handling | no contact stand before the ramp reopens |
| RADAR_OUTAGE | PHL TRACON/Newark 2025, ~90 s radar+radio loss | all agents down → independent planner + alarm |
| STAFFING_SHORTAGE | US ~3,500 controllers short (2025) | single runway, extra spacing, flaky agents, 20 s hard budget |
| RUNWAY_CLOSURE | disabled aircraft | closed runway never used |
| ROGUE_AGENTS | safety drill | unsafe ACCEPT overridden |
| STORM_FRONT_WAVE | arrivals compressed into remaining capacity | parallel negotiations on a stale snapshot; lock-protected ledger re-validates, re-plans, fuel priority; no wake conflict, no double-booked stand |

## Ops settings and automated weather watch (⚙️ Ops settings page, sidebar watch)
- **Settings wired into the safety checks and agents** (`ops_settings.py`, saved to `assets/ops_settings.json`):
  runway condition code RWYCC 6→0 (ICAO GRF, illustrative table: lower crosswind limit and extra spacing, 0 closes
  the runway), approach mode (Auto/Visual/ILS CAT I/CAT II-III spacing), crosswind and tailwind limits, minimum-fuel
  margin, maximum holding in severe weather, minimum connection time, ramp staffing (stretches turnarounds), remote
  stands on/off, crew overtime cap, negotiation time budget and default autonomy.
- **Weather watch** (`watch.py`): every N minutes across a watchlist (default SIN, ORD, ATL, DFW) it checks:
  thunderstorms (METAR TS/CB, TAF, forecast storm codes, CAPE); heavy rain; crosswind/tailwind per runway against the
  runway-condition-adjusted limits; gusts; IFR/LIFR (low-visibility procedures); wind shear; MOD/SEV turbulence and
  icing PIREPs; freezing precipitation; convective SIGMETs; and failed or stale feeds.
  It only notifies on change (NEW / ESCALATED / CLEARED) to avoid alarm fatigue. Alerts go to sidebar toasts and
  sound, the watch board, Slack/Teams webhook, Mac notifications and **phone push via [ntfy](https://ntfy.sh)**.
  The app runs it on a timer, or run it standalone with `python watch.py` (`--once` for a single pass).

## Realtime API for a custom dashboard (`api.py`)
`uvicorn api:app --reload --port 8000` serves REST for actions and one WebSocket (`/ws`) that pushes everything live:
weather-watch alerts (change-only), each negotiation step with every agent's **thinking summary** (Claude's
summarised reasoning, `thinking.display: "summarized"`), pending decisions, alarms and emergencies.
`docs/REACT_DASHBOARD_PROMPT.md` is a ready-to-paste prompt that makes Claude build an Apple-style React dashboard
against this exact contract.

## Guided demo and voice
- **Ops settings → 🎬 Demo** sends one prompt to #ops-team that makes the Chief of Staff run the whole flow on one real DOT row: **UA1112, Dulles → Denver,
  6 May 2015** (row 115187; WEATHER_DELAY 37, LATE_AIRCRAFT_DELAY 26; real KDEN thunderstorm with lightning, MOD
  turbulence PIREP, runways closed, stand conflict, tight fuel). The steps are: meet the team · brief the Chief with the
  DOT row · cross-domain handoff · bounded negotiation with stakeholder votes · rogue agents blocked · AI outage
  fallback · weather alert and notifications · MAYDAY · log. Run it with live Claude agents (about $1-2) or
  deterministic (free).
- **Voice:** in CHATS click 🎙️ Enable voice, allow the microphone, and speak. Speech is transcribed locally with Whisper
  (faster-whisper, offline). Starting with an agent's name ("ATC…", "Captain…") works like an @mention.

## Failure analysis (🔍 FAILURE ANALYSIS page, `failures.py`)
Every negotiation that ended in the deterministic planner or a human override, read from the consensus logs:
the fallback rate, the primary cause (which hard limit kept blocking plans), the deadlock point (the last plan and
exactly what it broke), which domains collided, and the cost of failure (USD and CO2 at 3.16 kg per kg of fuel)
against the best safe plan the AI produced. A **what-if replay** reruns the same scenario with levers an operator
controls (extra fuel uplift, ramp staffing, a free stand). Regulatory minimums are never relaxed in a replay.

## Data
- `data/flights.csv` is the US DOT 2015 extract. Each scenario starts from one of its rows.
- `data/challenge2_operational_constraints.xlsx` is the organisers' limits workbook, loaded by `limits.py`. Its airport
  figures are synthetic, as the workbook says. The consensus log records its SHA-256 and the airport's limits.

| Workbook rule | Where it applies |
|---|---|
| `active_runways`, `max_runway_ops_per_hour` | Runways in the scenario; background traffic; hard check `runway_capacity` |
| `gates`, `spare_gates_at_any_time` | Contact stands; how many are free when the aircraft is due |
| `min_turnaround_min` | Floor on the turnaround time |
| Wake table (Super, Heavy, Large, Small) | Hard check `wake_separation` |
| Final reserve, contingency, holding burn, spare fuel | Fuel on board, minimum landing fuel, hard check `fuel_reserve` |
| Flight crew max duty | Hard check `flight_crew_duty` |
| Ground crew shift, overtime multiplier | Shift changes; overtime is a soft check and a cost, not a hard limit |
| Gate change notice | Hard check `gate_change_notice` |
| One turn = each agent sends one message; 3 turns, then the fallback | `negotiation.py`, per round of debate |

**What this team added or assumed.** The workbook asks teams to state these in the demo:
- **Turn limit (changed):** the first consensus must come within the workbook's 3 turns. If the stakeholders then reject
  it, a second round gets up to 3 more, so a flight can take up to 6 turns. Ratification itself is not a turn.
- **Stricter than the workbook:** the alternate's fuel (ALTN) is protected as well as the final reserve; the 5%
  contingency is protected rather than available for holding; IMC adds 1 min of spacing and lowers the background
  arrival rate; holding in severe PIREPs is capped at 10 min; crosswind and tailwind limits.
- **Wake table:** the workbook's `min_separation_nm` is converted to whole minutes at a 140 kt final approach, rounded
  up. Its `min_time_s` column is for departures, which are not planned here.
- **Capacity:** half of `max_runway_ops_per_hour` is left for departures when generating background arrivals. The hard
  check counts arrivals only, because departures are not modelled.
- **Stands:** at most 10 contact stands are modelled, plus a remote bay `R1`. Which stands are busy is simulated.
- **Crew:** flight crew report 60 min before departure, and duty is estimated from the tail's legs because DOT has no
  crew data. Ground shifts change at 06:00, 14:00 and 22:00 local.
- **Gate-change notice:** counted from when the negotiation opens (simulated, 10-40 min before ELDT) to the landing time.
- **Aircraft:** DOT has no aircraft type, so distance >= 2000 mi means Heavy and anything shorter means Large. Super and
  Small only appear in simulated background traffic. Cruise burn and speed, used for trip and alternate fuel, are ours.
- `eda.py` produces `eda/EDA_REPORT.md` and `eda/calibration.json` (taxi-in, arrival rates, turnaround). Key findings: the file
  is pre-filtered, all rows already meet the rule, October is missing, 88% of rows are late-aircraft only, and inbound delay
  propagates to the next leg with correlation 0.76.
- `weather.py` handles weather in plain code with no LLM. It uses the IEM METAR and PIREP archives plus Open-Meteo ERA5 for
  the flight's real date, aviationweather.gov for live data, and OurAirports for runways. Every payload is cached with a
  SHA-256 and recorded in the consensus log.
- `weather_poller.py` refreshes the live cache on an interval.

## Safety and control
- Hard checks (`constraints.py`): wake separation (+1 min IMC), runway capacity, runway closure, crosswind ≤ 33 kt /
  tailwind ≤ 10 kt, FRF + ALTN + contingency, flight crew duty, max 10 min holding in SEV PIREPs, stand size and
  availability, gate-change notice.
- Real-ID enums: agents can only name runways and stands that exist in the scenario. Plan numbers make stale replies
  get discarded.
- Autonomy tiers: AUTONOMOUS, SUPERVISED and STEP_BY_STEP. Decision points cover stakeholder votes, turn review,
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
python -m unittest discover tests         # 53 offline tests
```
