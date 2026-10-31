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
                       weather_card,
                       ratification_form, render_emergency, render_event, render_outcome, timeline)

AUTONOMY_HELP = {
    "AUTONOMOUS": "Never pauses. The AI agents vote on the agreed plan and every decision point takes its safe "
                  "default; alarms still go out.",
    "SUPERVISED": "The controller, pilot and ground lead vote on the agreed plan. Also pauses on risk: deadlock, agent "
                  "failure or timeout, fallback.",
    "STEP_BY_STEP": "Everything in Supervised, and pauses after every turn.",
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
    st.session_state.update(cr_scen_key=scen_key, sess=None, events=[], pending=None, ratify=None, outcome=None,
                            choice=None, played=set(), cr_emergency=None, cr_steered=None, cr_halted=False,
                            cr_running=False)
if st.session_state.get("cr_steered") is not None:
    s = st.session_state.cr_steered  # the flight as updated by the duty officer's steering

# ============================================================================
# Header
# ============================================================================
hero(compact=True)
fids_board(s, st.session_state.get("outcome"), emergency=st.session_state.get("cr_emergency"),
           running=st.session_state.get("sess") is not None,
           pending=st.session_state.get("pending") is not None or st.session_state.get("ratify") is not None)
m = st.columns(6)
m[0].metric("Flight", s.flight, f"{s.origin} → {s.destination}", delta_color="off")
m[1].metric("SIBT", s.clock(0), s.date, delta_color="off")
m[2].metric("WEATHER_DELAY", f"{s.weather_delay_min} min")
m[3].metric("LATE_AIRCRAFT", f"{s.late_aircraft_delay_min} min")
m[4].metric("ELDT", s.clock(s.eta_min), f"slot {s.eta_min}", delta_color="off")
m[5].metric("Latest safe TLDT", s.clock(s.latest_safe_landing), "FRF+ALTN+cont.", delta_color="off")
weather_card(s)

# Stress tests, batch evaluation and EDA stay available from the command line (stress.py, batch_eval.py, eda.py).
tab_neg, tab_wx, tab_mon = st.tabs(["🛫 Negotiation", "🌦️ Weather & provenance", "📈 Monitoring"])


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

    b1, b2, _ = st.columns([1.2, 0.8, 4])
    b1.button("▶ Start negotiation", type="primary", disabled=use_claude and not key_ok,
              on_click=lambda: st.session_state.update(cr_start=True))
    if b2.button("↺ Reset"):
        st.session_state.update(sess=None, events=[], pending=None, ratify=None, outcome=None, choice=None, played=set(),
                                cr_emergency=None, cr_steered=None, cr_halted=False, cr_running=False)
        st.rerun()
    req_m = st.session_state.pop("mayday_request", None)  # from the sidebar's confirmed MAYDAY
    if req_m:
        plan = plan_emergency(s, req_m["kind"], req_m["reason"])
        dispatch(Alarm(code="MAYDAY", severity="ALARM", title=f"{KIND_LABEL[req_m['kind']]} {s.flight}",
                       detail=f"Divert {plan.divert_to}" if plan.divert_to else
                       f"RWY {plan.proposal.runway} {plan.landing_clock}, stand {plan.proposal.gate}",
                       flight=s.flight, turn=0))
        st.session_state.update(sess=None, pending=None, ratify=None, cr_emergency=plan, cr_em_live=True,
                                cr_running=False, cr_halted=False)
        st.rerun()
    if st.session_state.get("cr_emergency") is not None:
        render_emergency(s, st.session_state.cr_emergency, live=st.session_state.pop("cr_em_live", False),
                         sound=sound)

    def build_agents():
        if use_claude:
            overrides = {r: m_ for r, m_ in per_role.items() if m_ != "(same)"}
            return make_agents(model=model, effort=effort, moderator=use_moderator, models=overrides,
                               timeout_s=float(timeout_s))
        return make_engine_agents(kind, "", "")

    # 1. A fresh start creates the session first, so the live panel below is already on screen while agents talk.
    if st.session_state.pop("cr_start", False):
        st.session_state.update(sess=NegotiationSession(s, build_agents(), Autonomy(autonomy)), events=[], pending=None,
                                ratify=None, outcome=None, choice=None, cr_halted=False, cr_running=True)

    # 2. Live control panel: ABOVE the conversation, on screen at all times.
    live = st.session_state.get("sess") is not None and st.session_state.outcome is None
    if True:
        with st.container(border=True):
            if st.session_state.get("cr_halted"):
                st.markdown("⏸️ **Negotiation stopped.** Tell the team what changed and send it as **Steer**, or:")
                h1, h2, _ = st.columns([1, 1.7, 3])
                h1.button("▶ Resume", key="cr_resume",
                          on_click=lambda: st.session_state.update(cr_halted=False, cr_running=True))
                h2.button("⛔ Deterministic layer decides", key="cr_override_now",
                          on_click=lambda: st.session_state.update(cr_override_go=True))
            elif live:
                c1, c2 = st.columns([3, 1], vertical_alignment="center")
                c1.markdown("🟢 **Live negotiation.** Something changed? Stop it, or type it below and the team "
                            "will re-plan.")
                c2.button("⏹ Stop", key="cr_stop", type="primary", width="stretch",
                          on_click=lambda: st.session_state.update(cr_halted=True, cr_running=False))
            else:
                c1, c2 = st.columns([3, 1], vertical_alignment="center")
                c1.markdown("⚪ **No negotiation running.** Something changed? Type it below: the flight is updated "
                            "and a negotiation starts with the new information.")
                c2.button("⏹ Stop", key="cr_stop_idle", width="stretch", disabled=True)
            with st.form("cr_steer_form", clear_on_submit=True, border=False):
                text = st.text_input("What changed?", placeholder="e.g. Runway 22L closed for 40 minutes, disabled "
                                     "aircraft · stand B2 lost to a tow · fuel leak, 10 minutes less fuel")
                mode = st.segmented_control("Send as", ["Steer", "Override"], default="Steer",
                                            format_func={"Steer": "🧭 Steer: Chief synthesizes & restarts",
                                                         "Override": "⛔ Override: deterministic layer decides"}.get)
                if st.form_submit_button("Send to the team", type="primary") and text.strip():
                    if len(os.environ.get("ANTHROPIC_API_KEY", "")) < 40:
                        st.error("Steering needs the Chief of Staff (Claude): add your API key to .env.")
                    else:
                        st.session_state.cr_steer_req = (text.strip(), mode or "Steer")
                        st.session_state.update(cr_halted=True, cr_running=False)  # freeze the negotiation now

    # 3. The conversation so far.
    for k, d in st.session_state.events:
        render(k, d)
    status = st.empty()

    def pump(choice=None):
        sess: NegotiationSession = st.session_state.sess
        st.session_state.update(cr_running=True, cr_halted=False)
        ev = sess.advance(choice)
        while True:
            k, d = ev
            if k == "thinking":
                status.info(" · ".join(NAMES[a] for a in d["agents"]) + " working…")
            elif k == "metric":
                pass
            else:
                st.session_state.events.append(ev)
                render(k, d, live=True)
            if k in ("ratification", "decision", "outcome"):
                st.session_state.cr_running = False
            if k == "ratification":
                status.empty()
                st.session_state.ratify = d
                return
            if k == "decision":
                status.empty()
                st.session_state.pending = d
                return
            if k == "outcome":
                status.empty()
                st.session_state.outcome = d
                return
            ev = sess.advance()

    def note(kind_: str, text_: str) -> None:
        ev = ("system", {"turn": 0, "kind": kind_, "text": text_})
        st.session_state.events.append(ev)
        render(*ev, live=True)

    def restart(new_s, agents, autonomy_: str, directive: str, label: str) -> None:
        sess = NegotiationSession(new_s, agents, Autonomy(autonomy_))
        if directive:  # on every agent's bus, every turn (the constraint ledger is always shown)
            sess.ledger[("CHIEF", "operator_directive")] = f"CHIEF OF STAFF DIRECTIVE (from the duty officer): {directive}"
            sess.log.add(0, "HUMAN", "operator_directive", {"directive": directive})
        st.session_state.update(sess=sess, pending=None, ratify=None, outcome=None, cr_halted=False)
        note("negotiation_restarted", label)
        pump()

    def steer_live(text_: str, override: bool) -> None:
        """Chief synthesizes -> code firewall -> new facts applied to the flight -> restart from turn 1."""
        import steer
        st.session_state.update(cr_running=False, cr_halted=True)
        note("negotiation_stopped", f"Duty officer: “{text_}”")
        chief = make_agents(model=model if use_claude else DEFAULT_MODEL,
                            effort=effort if use_claude else "medium")["MODERATOR"]
        record = "\n".join(f"{k}: {getattr(d, 'rationale', '') or (d.get('text', '') if isinstance(d, dict) else '')}"
                           for k, d in st.session_state.events[-20:])
        with st.spinner("Chief of Staff is synthesizing the new information…"):
            directive = steer.synthesize(chief, s, text_, record)
        if directive.emergency != "NONE":
            st.session_state.mayday_request = {"kind": directive.emergency, "reason": directive.summary}
            return
        accepted, refused, ignored = steer.firewall(s, directive)
        if refused:
            with st.spinner("The safety firewall refused part of it: the Chief is explaining…"):
                r = steer.explain_refusal(chief, text_, refused)
            note("firewall_refused", f"**{r.header}.** {r.explanation} Safe alternative you can send: "
                                     f"_{r.safe_alternative}_ The negotiation stays stopped.")
            return
        new_s = steer.apply(s, accepted, directive.summary)
        st.session_state.cr_steered = new_s
        applied = [steer.describe(new_s, e) for e in accepted]
        text_d = directive.summary + (" Applied: " + "; ".join(applied) + "." if applied else "") + "".join(
            f" {g.agent}: {g.instruction}" for g in directive.guidance)
        note("operator_directive", f"{directive.summary}"
             + (f" **Now enforced by the safety checks:** {'; '.join(applied)}." if applied else "")
             + (f" _Ignored: {'; '.join(ignored)}_" if ignored else ""))
        if override:
            restart(new_s, make_engine_agents("Deterministic", "", ""), "AUTONOMOUS", text_d,
                    "Override: the deterministic layer decides on the updated flight (no AI).")
        else:
            restart(new_s, build_agents(), autonomy, text_d, "Negotiation restarted from turn 1 with the new information.")

    # 4. Requests, processed after the conversation so new messages appear in order.
    steer_req = st.session_state.pop("cr_steer_req", None)
    if steer_req:
        steer_live(steer_req[0], override=steer_req[1] == "Override")
        st.rerun()
    if st.session_state.pop("cr_override_go", False):
        restart(s, make_engine_agents("Deterministic", "", ""), "AUTONOMOUS", "",
                "Override: the deterministic layer decides (no AI).")
        st.rerun()
    if st.session_state.get("choice") is not None and st.session_state.get("sess") is not None:
        choice, st.session_state.choice = st.session_state.choice, None
        pump(choice)
        st.rerun()
    # A fresh start, or a negotiation interrupted by any click, runs (on) from here.
    if (st.session_state.get("sess") is not None and st.session_state.get("cr_running")
            and not st.session_state.get("cr_halted") and st.session_state.outcome is None
            and st.session_state.get("pending") is None and st.session_state.get("ratify") is None):
        pump()
        st.rerun()

    # 5. Cards that need the duty officer, and the outcome.
    if st.session_state.get("ratify") is not None:
        votes = ratification_form(st.session_state.ratify, "cr_ratify")
        if votes is not None:
            st.session_state.update(choice=votes, ratify=None)
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
        index = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
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
    sources = {
        "CONTROLS negotiation": st.session_state.sess.metrics if st.session_state.get("sess") else [],
        "CHATS negotiation": st.session_state.oc_session.metrics if st.session_state.get("oc_session") else [],
        "CHATS conversations": st.session_state.get("oc_metrics", []),
    }
    available = [k for k, v in sources.items() if v]
    picked_src = st.segmented_control("Source", ["All", *sources], default="All", key="mon_src")
    metrics = [m for k, v in sources.items() for m in v] if picked_src in (None, "All") else sources[picked_src]
    st.caption("Calls recorded this session: " + (", ".join(f"{k} {len(sources[k])}" for k in available) or "none yet"))
    if not metrics:
        st.info("No Claude calls recorded yet. Chat with the agents in CHATS or start a negotiation with Claude agents; "
                "deterministic runs make no API calls, so they show nothing here.")
    else:
        u = summarise(metrics)
        c = st.columns(5)
        c[0].metric("Calls", u["calls"], f"{u['failed_calls']} failed", delta_color="inverse")
        c[1].metric("Input tokens", f"{u['input_tokens']:,}")
        c[2].metric("Output tokens", f"{u['output_tokens']:,}", "incl. thinking", delta_color="off")
        c[3].metric("Cost", f"${u['cost_usd']:.4f}")
        c[4].metric("Latency p50 / max", f"{u['latency_p50_s']}s", f"max {u['latency_max_s']}s", delta_color="off")
        mdf = pd.DataFrame([x.model_dump() for x in metrics])
        fig = go.Figure(go.Bar(x=[f"#{i + 1} {a}" for i, a in enumerate(mdf.agent)], y=mdf.latency_s,
                               marker_color=["#d62728" if not ok else "#4c78a8" for ok in mdf.ok]))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Latency per call (s)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(mdf, hide_index=True, width="stretch")
        st.caption("Pricing table in monitoring.py. Output tokens include adaptive-thinking tokens, which are billed "
                   "as output. Agents see a compact bus (constraint ledger + latest turn), not the full transcript.")
