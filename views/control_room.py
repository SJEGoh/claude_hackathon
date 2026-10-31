"""Control room: dashboard view of one negotiation, weather provenance, monitoring, batch runs and EDA."""
import json
import os
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config  # noqa: F401  (loads .env)
from agents import DEFAULT_MODEL, make_agents
from agents.atc_agent import ATCAgent
from agents.ground_agent import GroundAgent
from agents.pilot_agent import PilotAgent
from escalation import duty_phone
from monitoring import summarise
from negotiation import MAX_TURNS, Autonomy, NegotiationSession
import costs
from emergency import KIND_LABEL, plan_emergency
from escalation import Alarm, dispatch
from ui_common import (AVATARS, ENGINES, NAMES, ROOT, fids_board, flights, get_scenario, hero, make_engine_agents,
                       render_emergency, render_event, render_outcome, timeline)

AUTONOMY_HELP = {
    "AUTONOMOUS": "Never pauses. Every decision point takes its safe default; alarms still go out.",
    "SUPERVISED": "Pauses only on risk: an agreed plan near a limit, deadlock, agent failure or timeout, fallback.",
    "STEP_BY_STEP": "Pauses after every turn and before anything is committed.",
}


def render(kind, data, live=False):
    render_event(s, kind, data, live=live, sound=sound, autonomy=autonomy)


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.header("Agents")
    kind = st.radio("Agent engine", ENGINES)
    use_claude = kind == ENGINES[0]
    key_ok = len(os.environ.get("ANTHROPIC_API_KEY", "")) > 40
    if use_claude:
        st.caption("✅ API key loaded from .env" if key_ok else "⚠️ No API key: add it to .env")
        model = st.selectbox("Model (all roles)", [DEFAULT_MODEL, "claude-sonnet-5-5", "claude-haiku-5-5"])
        with st.expander("Per-role model override"):
            per_role = {r: st.selectbox(r, ["(same)", DEFAULT_MODEL, "claude-sonnet-5-5", "claude-haiku-5-5"], key=f"m_{r}")
                        for r in ("ATC", "PILOT", "GROUND", "MODERATOR")}
        effort = st.select_slider("Effort", ["low", "medium", "high"], value="medium", help="medium shows the agents' thinking summaries; low is cheaper and often skips thinking")
        timeout_s = st.slider("Per-call timeout (s)", 15, 180, 90)
        use_moderator = st.toggle("Chief of Staff moderator", value=True)

    st.header("Human oversight")
    autonomy = st.radio("Autonomy tier", [a.value for a in Autonomy], index=1,
                        format_func=lambda a: a.replace("_", " ").title())
    st.caption(AUTONOMY_HELP[autonomy])
    with st.expander("Escalation channels"):
        phone = st.text_input("Duty manager phone", value=duty_phone(), placeholder="+65…")
        hook = st.text_input("Alarm webhook URL (Slack/Teams)", value=os.environ.get("ESCALATION_WEBHOOK_URL", ""),
                             type="password")
        os.environ["DUTY_MANAGER_PHONE"], os.environ["ESCALATION_WEBHOOK_URL"] = phone, hook
        sound = st.toggle("Audible alarm", value=True)

    with st.expander("💲 Cost weights (USD)"):
        st.caption("Used ONLY to rank plans that already pass every hard check. Defaults and sources: costs.py.")
        w = costs.weights()
        new = {k: st.number_input(k.replace("_", " "), value=float(v), min_value=0.0, key=f"w_{k}")
               for k, v in w.model_dump().items()}
        c1, c2 = st.columns(2)
        if c1.button("Save weights"):
            costs.save_weights(costs.Weights(**new))
            st.rerun()
        if c2.button("Defaults"):
            costs.save_weights(costs.Weights())
            st.rerun()

    st.header("Disruption (US DOT 2015)")
    df = flights()
    c1, c2 = st.columns(2)
    bucket = c1.selectbox("Bucket", ["all", "late_aircraft", "weather", "both"])
    split = c2.selectbox("Split", ["all", "dev", "test"])
    view = df if bucket == "all" else df[df.BUCKET == bucket]
    view = view if split == "all" else view[view.SPLIT == split]
    labels = {i: f"{r.AIRLINE}{r.FLIGHT_NUMBER} {r.ORIGIN_AIRPORT}→{r.DESTINATION_AIRPORT} {r.MONTH:02d}-{r.DAY:02d} "
                 f"· WX {int(r.WEATHER_DELAY)} · LATE {int(r.LATE_AIRCRAFT_DELAY)}" for i, r in view.iterrows()}
    pick = st.selectbox(f"Flight ({len(view)})", list(labels), format_func=labels.get)
    weather_mode = st.radio("Weather", ["HISTORICAL", "LIVE", "OFF"], horizontal=True,
                            help="HISTORICAL: IEM METAR/PIREP archive + Open-Meteo for the flight's real date. "
                                 "LIVE: today's aviationweather.gov data (what-if). OFF: no weather.")

s = get_scenario(pick, weather_mode)
scen_key = (pick, weather_mode)
if st.session_state.get("cr_scen_key") != scen_key:
    st.session_state.update(cr_scen_key=scen_key, sess=None, events=[], pending=None, outcome=None, choice=None,
                            played=set(), cr_emergency=None)

# ============================================================================
# Header
# ============================================================================
hero(f"Control room · AUTONOMOUS SG 2026 · Challenge #2 · three independent specialists, a Chief of Staff, max "
     f"{MAX_TURNS} turns, deterministic safety engine, human oversight tiers", compact=True)
fids_board(s, st.session_state.get("outcome"), emergency=st.session_state.get("cr_emergency"),
           running=st.session_state.get("sess") is not None,
           pending=st.session_state.get("pending") is not None)
m = st.columns(6)
m[0].metric("Flight", s.flight, f"{s.origin} → {s.destination}", delta_color="off")
m[1].metric("SIBT", s.clock(0), s.date, delta_color="off")
m[2].metric("WEATHER_DELAY", f"{s.weather_delay_min} min")
m[3].metric("LATE_AIRCRAFT", f"{s.late_aircraft_delay_min} min")
m[4].metric("ELDT", s.clock(s.eta_min), f"slot {s.eta_min}", delta_color="off")
m[5].metric("Latest safe TLDT", s.clock(s.latest_safe_landing), "FRF+ALTN", delta_color="off")
(st.warning if s.weather_at_destination else st.info)(s.weather_attribution, icon="🌦️")

tab_neg, tab_wx, tab_mon, tab_stress, tab_batch, tab_eda = st.tabs(
    ["🛫 Negotiation", "🌦️ Weather & provenance", "📈 Monitoring", "🔥 Stress tests", "🧪 Batch evaluation", "🔎 EDA"])


# ============================================================================
# Negotiation tab
# ============================================================================
with tab_neg:
    with st.expander("What each specialist privately knows (no agent sees another's column)"):
        cols = st.columns(3)
        for col, cls, key in [(cols[0], ATCAgent, "ATC"), (cols[1], PilotAgent, "PILOT"), (cols[2], GroundAgent, "GROUND")]:
            col.markdown(f"**{AVATARS[key]} {NAMES[key]}**")
            col.code(cls(None).private_brief(s), language=None, wrap_lines=True)
    chart = st.empty()
    out = st.session_state.outcome
    chart.plotly_chart(timeline(s, out.proposal if out else None, (out.reference.proposal if out else None)),
                       width="stretch")

    b1, b2, b3, b4 = st.columns([1.2, 0.8, 1.2, 1.4])
    start = b1.button("▶ Start negotiation", type="primary", disabled=use_claude and not key_ok)
    if b2.button("↺ Reset"):
        st.session_state.update(sess=None, events=[], pending=None, outcome=None, choice=None, played=set(),
                                cr_emergency=None)
        st.rerun()
    em_kind = b3.selectbox("Emergency", list(KIND_LABEL), format_func=KIND_LABEL.get, label_visibility="collapsed")
    if b4.button("🚨 DECLARE MAYDAY", help="Abandon negotiation; deterministic emergency plan in milliseconds."):
        plan = plan_emergency(s, em_kind)
        dispatch(Alarm(code="MAYDAY", severity="ALARM", title=f"{KIND_LABEL[em_kind]} {s.flight}",
                       detail=f"Divert {plan.divert_to}" if plan.divert_to else
                       f"RWY {plan.proposal.runway} {plan.landing_clock}, stand {plan.proposal.gate}",
                       flight=s.flight, turn=0))
        st.session_state.update(sess=None, pending=None, cr_emergency=plan, cr_em_live=True)
        st.rerun()
    if st.session_state.get("cr_emergency") is not None:
        render_emergency(s, st.session_state.cr_emergency, live=st.session_state.pop("cr_em_live", False),
                         sound=sound)

    for k, d in st.session_state.events:
        render(k, d)

    status = st.empty()

    def pump(choice=None):
        sess: NegotiationSession = st.session_state.sess
        ev = sess.advance(choice)
        while True:
            k, d = ev
            if k == "thinking":
                status.info(" · ".join(f"{AVATARS[a]} {NAMES[a]}" for a in d["agents"]) + " working…")
            elif k == "metric":
                pass
            else:
                st.session_state.events.append(ev)
                render(k, d, live=True)
            if k == "decision":
                status.empty()
                st.session_state.pending = d
                return
            if k == "outcome":
                status.empty()
                st.session_state.outcome = d
                return
            ev = sess.advance()

    if start:
        if use_claude:
            overrides = {r: m_ for r, m_ in per_role.items() if m_ != "(same)"}
            agents = make_agents(model=model, effort=effort, moderator=use_moderator, models=overrides,
                                 timeout_s=float(timeout_s))
        else:
            agents = make_engine_agents(kind, "", "")
        st.session_state.update(sess=NegotiationSession(s, agents, Autonomy(autonomy)), events=[], pending=None,
                                outcome=None)
        pump()
        st.rerun()
    elif st.session_state.choice is not None and st.session_state.sess is not None:
        choice, st.session_state.choice = st.session_state.choice, None
        pump(choice)
        st.rerun()

    req = st.session_state.pending
    if req is not None:
        colour = {"INFO": "blue", "WARNING": "orange", "ALARM": "red"}[req.severity]
        with st.container(border=True):
            st.markdown(f"### :{colour}[👤 Your decision · {req.title}]")
            st.caption(f"{req.code} · turn {req.turn} · tier {autonomy.replace('_', ' ').lower()} · "
                       f"autonomous default would be **{req.default}**")
            for line in req.briefing:
                st.markdown(f"- {line}")
            for opt in req.options:
                c1, c2 = st.columns([1, 3])
                if c1.button(opt.label, key=f"opt_{req.code}_{req.turn}_{opt.id}",
                             type="primary" if opt.id == req.default else "secondary"):
                    st.session_state.update(choice=opt.id, pending=None)
                    st.rerun()
                c2.caption(opt.description)
            if req.severity == "ALARM" and duty_phone():
                st.link_button(f"📞 Call duty manager {duty_phone()}", f"tel:{duty_phone()}")

    if st.session_state.outcome:
        render_outcome(s, st.session_state.outcome)

# ============================================================================
# Weather tab
# ============================================================================
with tab_wx:
    wx = s.weather
    if wx is None:
        st.info("Weather is OFF or no source answered for this airport.")
    else:
        st.subheader(f"{wx.airport_icao} · {wx.mode.lower()} · valid {wx.valid_local} local")
        st.markdown(f"**{wx.summary()}**")
        st.code(f"METAR {wx.metar or 'n/a'}" + (f"\n{wx.taf}" if wx.taf else ""), language=None, wrap_lines=True)
        c1, c2 = st.columns(2)
        c1.markdown("**Runway wind components (in use)**")
        c1.dataframe(pd.DataFrame([{"runway": r.id, "heading": r.heading_true, "head kt": r.headwind_kt,
                                    "cross kt": r.crosswind_kt, "tail kt": r.tailwind_kt,
                                    "within limits": r.wind_within_limits, "status": r.closure_reason}
                                   for r in s.runways]), hide_index=True, width="stretch")
        c2.markdown("**Open-Meteo (ERA5) at the hour**")
        c2.json(wx.open_meteo or {"note": "not used in live mode"})
        st.markdown(f"**PIREPs ≤ FL180 within 100 NM, ±3 h ({len(wx.pireps)})**")
        if wx.pireps:
            st.dataframe(pd.DataFrame([p.model_dump() for p in wx.pireps]), hide_index=True, width="stretch")
        st.markdown("**Source provenance** (raw payloads cached under `weather_cache/` with SHA-256)")
        st.dataframe(pd.DataFrame([x.model_dump() for x in wx.sources]), hide_index=True, width="stretch")

    @st.fragment(run_every=60)
    def live_panel():
        st.markdown("---\n**Live feed (aviationweather.gov, refreshed by `weather_poller.py`)**")
        idx_path = ROOT / "weather_cache" / "live_index.json"
        index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
        if st.button(f"Poll {s.destination} now"):
            import weather_poller
            index = weather_poller.poll([s.destination])
        entry = index.get(s.destination)
        if entry:
            st.success(f"{entry['summary']}  \npolled {entry['polled_at']} UTC")
            if entry.get("notes"):
                st.caption("; ".join(entry["notes"]))
        else:
            st.caption(f"No live poll for {s.destination} yet. Run `python weather_poller.py --airports "
                       f"{s.destination}` or press the button.")
    live_panel()

# ============================================================================
# Monitoring tab
# ============================================================================
with tab_mon:
    sess = st.session_state.sess
    metrics = sess.metrics if sess else []
    if not metrics:
        st.info("Run a negotiation with Claude agents to see per-call telemetry.")
    else:
        u = summarise(metrics)
        c = st.columns(5)
        c[0].metric("Calls", u["calls"], f"{u['failed_calls']} failed", delta_color="inverse")
        c[1].metric("Input tokens", f"{u['input_tokens']:,}")
        c[2].metric("Output tokens", f"{u['output_tokens']:,}", "incl. thinking", delta_color="off")
        c[3].metric("Cost", f"${u['cost_usd']:.4f}")
        c[4].metric("Latency p50 / max", f"{u['latency_p50_s']}s", f"max {u['latency_max_s']}s", delta_color="off")
        mdf = pd.DataFrame([x.model_dump() for x in metrics])
        fig = go.Figure(go.Bar(x=[f"T{t} {a}" for t, a in zip(mdf.turn, mdf.agent)], y=mdf.latency_s,
                               marker_color=["#d62728" if not ok else "#4c78a8" for ok in mdf.ok]))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Latency per call (s)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(mdf, hide_index=True, width="stretch")
        st.caption("Pricing table in monitoring.py. Output tokens include adaptive-thinking tokens, which are billed "
                   "as output. Agents see a compact bus (constraint ledger + latest turn), not the full transcript.")

# ============================================================================
# Batch evaluation tab
# ============================================================================
with tab_batch:
    st.markdown("Stratified by disruption bucket, split 80/20 dev/test by a stable hash. Tune prompts on **dev**, "
                "report on **test**.")
    st.code("python batch_eval.py --agents rules --n 5          # free baseline\n"
            "python batch_eval.py --split test --n 4 --workers 3 # Claude agents (~$0.2-0.5 per flight)", language="bash")
    reports = sorted((ROOT / "eval").glob("*.md"), reverse=True)
    if reports:
        chosen = st.selectbox("Saved runs", reports, format_func=lambda p: p.stem)
        st.markdown(chosen.read_text())
        csv = chosen.with_suffix(".csv")
        if csv.exists():
            st.dataframe(pd.read_csv(csv), hide_index=True, width="stretch")
    else:
        st.info("No batch runs yet.")

# ============================================================================
# EDA tab
# ============================================================================
with tab_stress:
    import stress
    st.markdown("Real-world ATM failure modes replayed against the live engine. Every case checks the same "
                "invariants: **no crash · never an unsafe agreed plan · ≤ 3 turns · inside the time budget · log "
                "hash chain intact**. Runs on rule-based agents (free, about 10 s).")
    case_info = {
        "NATS_RECORD": "UK NATS, Aug 2023: two different beacons sharing a name crashed the primary and the identical "
                       "backup. Here: corrupt and ambiguous records are quarantined; the rest of the batch continues.",
        "LIGHTNING_RAMP": "Lightning halts ramp work (HKIA: within 1 km; Changi suspends baggage handling). Aircraft "
                          "land but cannot reach contact stands.",
        "RADAR_OUTAGE": "PHL TRACON / Newark, Apr-May 2025: ~90 s radar and radio loss over a single feed. Here: every "
                        "AI agent is unreachable; the independent planner must take over and alarm.",
        "STAFFING_SHORTAGE": "US ATC ~3,500 controllers short (2025): single runway, extra spacing, slow and flaky "
                             "responders, 20 s hard budget.",
        "RUNWAY_CLOSURE": "A disabled aircraft closes the preferred runway for 45 minutes.",
        "ROGUE_AGENTS": "Adversarial agents push a plan that breaks the fuel limit.",
        "STORM_FRONT_WAVE": "Five real DOT arrivals at one hub negotiated in parallel on a stale snapshot, committed "
                            "through a lock-protected ledger that re-validates every plan (stale-state race).",
    }
    picked = st.multiselect("Cases", list(case_info), default=list(case_info))
    if st.button("▶ Run stress suite", type="primary"):
        with st.spinner("Running stress cases…"):
            st.session_state.stress = stress.run_suite(row=int(pick) if int(pick) < 190 else 40,
                                                       weather_mode="OFF", cases=picked)
    for r in st.session_state.get("stress", []):
        icon = "✅" if r.passed else "❌"
        with st.expander(f"{icon} **{r.case}** · {r.title} · {r.status} · {r.elapsed_s}s", expanded=not r.passed):
            st.caption(case_info.get(r.case, r.basis) + (f" [source]({r.source})" if r.source else ""))
            st.dataframe(pd.DataFrame([i.model_dump() for i in r.invariants]), hide_index=True, width="stretch")
            for n in r.notes:
                st.caption(n)

with tab_eda:
    report = ROOT / "eda" / "EDA_REPORT.md"
    if not report.exists():
        st.info("Run `python eda.py` first.")
    else:
        for chunk in re.split(r"(!\[[^\]]*\]\([^)]+\))", report.read_text()):
            img = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", chunk)
            if img:
                st.image(str(ROOT / "eda" / img.group(2)), caption=img.group(1))
            elif chunk.strip():
                st.markdown(chunk)
