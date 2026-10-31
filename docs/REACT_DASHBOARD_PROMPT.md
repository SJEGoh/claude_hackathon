You are a senior product designer and frontend engineer who has shipped software at Apple. Build a production-quality
React dashboard for an airport operations system, designed and coded to the standard of Apple's own apps (think the
restraint of Weather, Flighty-level clarity, the calm of macOS Settings). It connects to a backend I already have.

# What the product is
"A-CDM Ops" is a control-room dashboard for one disrupted flight at a time. Three independent AI specialists negotiate
how to handle the disruption, chaired by a fourth:
- ATC (Arrival Manager): runway, landing time, wake separation, throughput
- Flight Crew (Captain): fuel reserves, turbulence, passenger connections
- AOCC (Turnaround Coordinator): stand/gate, crew overtime, turnaround
- Chief of Staff: moderates, summarises, proposes compromises; no vote
They negotiate in at most 3 turns. A deterministic safety engine checks every plan; if they don't agree, a non-AI
planner decides. A human duty officer (the user) can be asked to decide. Separately, an automated weather watch checks
airports for thunderstorms, rain, wind, visibility, wind shear, turbulence/icing, freezing precipitation, SIGMETs and
stale data feeds, and pushes alerts only when something changes. There is also a MAYDAY emergency path.

# Design language: think like Apple, not like "a dashboard template"
Principles to apply in every decision:
1. Clarity. The single most important thing on any screen is obvious within one second: the flight's status and,
   when it exists, the decision the user must make. Everything else steps back.
2. Deference. The UI never competes with the content. No decorative gradients, no glow, no emoji in the chrome, no
   borders on everything. Content sits on calm surfaces.
3. Depth through material, not decoration. Use translucent materials (backdrop blur + saturation) only for layers that
   float: the top alert pill, sheets, popovers, the sidebar. Base content is solid.
4. Hierarchy with type, weight and space before colour. Colour is reserved for meaning (status), never decoration.
5. Consistency. One spacing scale, one radius scale, one motion curve, used everywhere.

Concrete system:
- Typography: font-family `-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", Inter, system-ui,
  sans-serif`; tabular numbers (`font-variant-numeric: tabular-nums`) for every time, slot, kg and $ value.
  Scale (px/weight): Large Title 34/700, Title 1 28/700, Title 2 22/600, Headline 17/600, Body 17/400, Callout 16/400,
  Subheadline 15/400, Footnote 13/400, Caption 12/500 uppercase with +0.04em tracking for section labels. Tight
  negative tracking (-0.02em) on titles only.
- Colour: support light AND dark mode via `prefers-color-scheme` (default to the system). Use Apple system colours:
  blue #007AFF / dark #0A84FF (interactive, "working"), green #34C759 / #30D158 (agreed, safe), orange #FF9500 /
  #FF9F0A (warning, counter-proposal), red #FF3B30 / #FF453A (alarm, emergency, destructive), purple #AF52DE /
  #BF5AF2 (Chief of Staff), gray #8E8E93. Backgrounds: light #F5F5F7 base, #FFFFFF cards; dark #000000 base,
  #1C1C1E cards, #2C2C2E elevated. Labels: primary at full contrast, secondary at ~60%, tertiary at ~30%. Hairline
  separators: 0.5px rgba(60,60,67,0.29) light / rgba(84,84,88,0.6) dark, inset like iOS grouped lists. WCAG AA contrast.
- Geometry: 8pt grid (4 for fine tuning). Radii: 10 (controls), 14 (list groups), 20 (cards), 28 (sheets), fully
  rounded pills. Shadows almost invisible (e.g. 0 1px 2px rgba(0,0,0,.04), 0 8px 24px rgba(0,0,0,.06)), none in dark mode.
- Icons: lucide-react at 1.5 stroke, sized to the text they sit beside, monochrome (secondary label colour) unless
  they carry status.
- Motion: framer-motion springs (stiffness ~400, damping ~32), 150-250 ms, layout animations for list inserts and
  expanding rows. Respect `prefers-reduced-motion` (no movement, cross-fade only). Nothing loops forever except a
  subtle 1.2 s "working" shimmer on an agent that is thinking.
- Controls: segmented controls (autonomy tier), iOS-style switches (settings), inset grouped lists (alerts,
  settings), sheets that slide up for decisions (with dimmed backdrop), toasts as a "Dynamic Island" pill.
- Accessibility: every control keyboard-reachable with a visible focus ring (2px system blue, 2px offset);
  aria-live="assertive" for ALARMs, "polite" for other alerts; hit targets >= 44px; never rely on colour alone
  (pair every status colour with a word or icon).

# Layout (desktop first, responsive down to 390px wide)
Three-column macOS-style split view:
1. Sidebar (260px, translucent material): app title "A-CDM Ops" in Title 2; sections Flight, Team, Weather,
   Settings; under Team a compact roster of the 4 agents with avatar (image URL from /api/state, 28px, rounded 8),
   name, and live status word + dot (Idle · Thinking · Waiting for you). Bottom: connection status ("Live" green dot /
   "Reconnecting…").
2. Main column:
   - Flight header: large tabular "TLDT 20:55" style hero numbers with labels (SIBT · ELDT · TLDT · RWY · STAND), the
     route "DFW → PHL", date, and a single status capsule (Delayed / Negotiating / Awaiting you / Agreed / SOP plan /
     Diverted / MAYDAY). Below it one line of weather attribution in secondary text.
   - Negotiation thread (the heart of the app): an iMessage-calm vertical thread grouped by "Turn 1 of 3" section
     headers. Each agent message is a card: avatar, name, a small stance tag (Proposes / Accepts / Counters, coloured
     blue / green / orange), the plan as tabular "20:55 · 27L · A2", the rationale in Body text, and a disclosure row
     "Thinking" (chevron) that expands with a layout animation to show the agent's thinking summary in Subheadline
     secondary text. Published constraints collapse under "N constraints" (lock icon for hard ones). The Chief of
     Staff's briefs are full-width, purple-tinted (5% tint) cards with summary, open conflicts as a quiet list, and
     the bridge plan. System notes (consensus, safety override, stale reply, fallback) are centred Footnote lines
     with an icon, like iMessage timestamps. While agents work, show their avatars with the shimmer and
     "ATC and Flight Crew are thinking…".
   - Outcome card when finished: big status, the plan, cost in $ with "vs optimum", a tiny horizontal bar breakdown
     of the cost items, robustness as "Holds in 94% of weight scenarios", tokens/$ spent, log integrity check mark.
3. Inspector (320px): "Weather watch" inset grouped list of active alerts sorted ALARM > WARNING > INFO, each row:
   severity glyph, airport ICAO in mono, title, observed time; tap expands detail. Under it "Recent changes" (NEW /
   ESCALATED / CLEARED). Then a destructive-style "Declare MAYDAY" button (red text on red 10% tint) that opens a
   confirmation sheet with three choices (Fuel, Medical, Technical failure) and an optional reason field.

Overlays:
- Dynamic Island alert pill: top-centre, translucent, expands with a spring when a weather.alert, alarm or
  emergency arrives (icon + "WSSS · Thunderstorm at the field"), auto-collapses after 6 s for WARNING/INFO, stays
  until dismissed for ALARM. Queue multiple alerts.
- Decision sheet: when decision.required arrives, slide up a sheet titled with the decision title, the briefing as
  a short list, options as large stacked buttons (the default option is the filled primary button; "Hand over to the
  deterministic planner" style options are secondary), and the severity shown as a capsule. Submitting POSTs the
  choice and closes the sheet optimistically.
- Emergency takeover: on emergency.declared, the flight header turns into a red-tinted banner "MAYDAY · Technical
  failure · decided in 0.14 ms", and the main column shows three parallel checklist cards (ATC / Flight Crew / AOCC)
  with their action lists. Calm, not flashy: red is used once, for the banner capsule.

Notifications and sound (must work):
- A small "Notifications" switch in the sidebar footer. Turning it on is the user gesture that (a) calls
  `Notification.requestPermission()` and (b) unlocks audio by playing a silent Web Audio buffer. After that:
  ALARM events play a short two-tone chime generated with the Web Audio API (no audio files needed: 880 Hz then
  660 Hz, 120 ms each, soft attack/release, -12 dB) and, when the tab is hidden, show a system Notification
  (title "WSSS · Thunderstorm at the field", body = detail). WARNING shows the pill only, no sound. Respect a
  "Sounds" switch.
- Settings page has a "Send test alert" button that calls POST /api/alerts/test.

# Backend contract (already running at http://localhost:8000, make the base URL configurable via VITE_API_URL)
REST:
- GET  /api/state -> { flight: Flight|null, runs: Run[], alerts: Alert[], settings: Settings,
                       agents: { ATC|PILOT|GROUND|MODERATOR: { name, avatar } } }
  avatar is either an emoji or a path like "assets/avatars/default_atc.png" (load from `${API}/${avatar}`).
- GET  /api/flights?airport=PHL&bucket=any|late_aircraft|weather|both&min_weather_delay=0&limit=20
       -> [{ row_id, flight, origin, destination, date, scheduled_arrival, weather_delay_min,
             late_aircraft_delay_min, bucket }]
- POST /api/flights/{row_id}/load?weather=HISTORICAL|LIVE|OFF -> Flight
- POST /api/negotiations { autonomy: "AUTONOMOUS"|"SUPERVISED"|"STEP_BY_STEP", engine: "claude"|"rules"|"drill",
       effort: "low"|"medium"|"high" } -> { run_id }
- POST /api/negotiations/{run_id}/decision { option_id } -> { ok }
- POST /api/emergency { kind: "FUEL"|"MEDICAL"|"TECHNICAL", reason } -> EmergencyPlan
- GET/PUT /api/settings (partial PUT merges) · GET /api/alerts -> { active, recent } · POST /api/alerts/check ·
  POST /api/alerts/test · GET /api/events?since=<id> (replay)
WebSocket: ws://localhost:8000/ws. Every message: { id, type, ts, data }. On connect the server replays the last 50
events. Reconnect with exponential backoff (1 s → 30 s); on reconnect, call /api/state and /api/events?since=lastId.
Event types and data:
- flight.loaded: Flight = { flight, airline, tail, origin, destination, destination_name, date, sibt, eldt,
  latest_safe_landing, weather_delay_min, late_aircraft_delay_min, aircraft, weather_mode, weather,
  runways: [{ id, crosswind_kt, status }] }
- negotiation.started: { run_id, flight, autonomy, engine, effort }
- negotiation.turn: { run_id, turn }
- agent.working: { run_id, agents: ["ATC"] | ["PILOT","GROUND"] | ["MODERATOR"], turn }
- agent.message: { run_id, agent, turn, stance: "PROPOSE"|"ACCEPT"|"COUNTER", plan_id, answering_plan,
  proposal: { landing_slot_min, runway, gate, tldt }, rationale, concession,
  constraints: [{ name, value, hard }], thinking: string|null, metric: { latency_s, input_tokens, output_tokens, cost_usd } }
- chief.brief: { run_id, turn, summary, open_conflicts: string[], bridge_proposal: Plan|null,
  directives: [{ agent, instruction }], thinking: string|null }
- system.note: { run_id, turn, kind: "consensus"|"safety_override"|"safety_block"|"stale_reply"|"fallback"|..., text }
- validation: { run_id, turn, plan_id, domains: { ATC|PILOT|GROUND: { hard_ok, checks: [{ name, passed, hard, detail }] } } }
- alarm: { code, severity: "WARNING"|"ALARM", title, detail, flight, turn, channels }
- decision.required: { run_id, code, severity: "INFO"|"WARNING"|"ALARM", title, turn, briefing: string[],
  default, options: [{ id, label, description }] }
- decision.auto / decision.made: { run_id, code?, choice? / option_id, by }
- negotiation.outcome: { run_id, status: "CONSENSUS"|"FALLBACK"|"HUMAN_DECISION", plan_source, turns,
  plan: Plan (runway "DIVERT" means divert to plan.gate), cost_usd, optimum_cost_usd,
  cost_breakdown: { "PILOT: holding fuel": 120, ... }, robustness: 0..1|null, usage: { calls, cost_usd, ... },
  alarms, log_verified, elapsed_s }
- emergency.declared: { kind, label, reason, flight, plan: Plan|null, divert_to, landing_clock, in_block_clock,
  efob_kg, fuel_status, displaced: string[], actions: { ATC: string[], PILOT: string[], GROUND: string[] },
  decision_ms, channels }
- weather.alert: { at, kind: "NEW"|"ESCALATED"|"CLEARED", alert: { airport, check, severity, title, detail,
  observed }, channels, test? }
- weather.board: { alerts: Alert[] } (full current board after every check) · settings.updated: Settings

# Engineering
- Vite + React 18 + TypeScript + Tailwind CSS + framer-motion + lucide-react. No other UI kit.
- Files: src/main.tsx, src/App.tsx, src/lib/api.ts (typed REST client), src/lib/useLiveEvents.ts (WebSocket hook
  with reconnect, replay and a reducer that turns events into app state), src/lib/sound.ts (Web Audio chime +
  unlock), src/types.ts (types for every event above), src/components/{Sidebar, FlightHeader, Thread, AgentMessage,
  ChiefBrief, OutcomeCard, DecisionSheet, AlertIsland, WeatherList, EmergencyView, FlightPicker, SettingsView}.tsx,
  tailwind.config.ts with the tokens above as theme extensions, index.css with the CSS variables for light/dark.
- A "Demo data" mode (toggle in Settings, on automatically when the backend is unreachable) that replays a scripted
  stream of realistic events with timers, so the UI can be shown without the backend.
- Flight picker: a sheet with a search field and a short list from /api/flights; choosing one loads it. A primary
  "Start negotiation" button beside a segmented control for autonomy (Autonomous / Supervised / Step by step).
- Empty states are designed, not blank: e.g. "No flight loaded" with one primary action.
- No lorem ipsum; use the event data. No console errors. Give me the full contents of every file, then the exact
  commands to run it (npm create vite, install, npm run dev).
