"""Rendering helpers shared by the Ops chat and the Control room pages."""
import io
import math
import struct
import wave
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import profiles
from constraints import Proposal, cost_breakdown, plan_robustness
from emergency import KIND_LABEL, EmergencyPlan
from escalation import duty_phone
from negotiation import MAX_TURNS, STAKEHOLDERS
from scenario import Scenario, build_scenario, load_disrupted_flights, split_of

ROOT = Path(__file__).parent
STANCE = {"PROPOSE": "blue", "ACCEPT": "green", "COUNTER": "orange"}
ENGINES = ["Claude agents", "Deterministic"]  # the rogue-agent safety drill is a demo step, not an engine
STAKEHOLDER_TITLE = {"ATC": "Air traffic controller", "PILOT": "Pilot", "GROUND": "Ground operations lead"}


class _Profile(dict):
    """Live lookup into the customisable agent profiles (name or avatar)."""

    def __init__(self, field: str):
        super().__init__()
        self.field = field

    def __getitem__(self, agent):
        p = profiles.load()
        if self.field == "avatar":
            return profiles.avatar_for_streamlit(agent, p)
        return p.get(agent, {}).get("name", agent)

    def get(self, agent, default=None):
        return self[agent] if agent in profiles.load() else default


AVATARS = _Profile("avatar")
NAMES = _Profile("name")


def make_engine_agents(engine: str, model: str, effort: str, **kw) -> dict:
    from agents import make_agents
    from baseline_agents import make_drill_agents, make_rule_agents
    if engine.startswith(("Deterministic", "Rule")):
        return make_rule_agents()
    if engine.startswith(("Safety drill", "Drill")):
        return make_drill_agents()
    return make_agents(model=model, effort=effort, **kw)


# ============================================================================
# Airport look: theme CSS, hero banner, departure-board (FIDS) strip
# ============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500;700&family=VT323&display=swap');
html, body, [class*="css"], .stMarkdown, .stChatMessage { font-family: 'Inter', sans-serif; }
code, .stCode, pre { font-family: 'JetBrains Mono', monospace !important; }
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 800 !important; }
[data-testid="stSidebar"] { background: #0A0C0E; border-right: 1px solid #1F242B; }
[data-testid="stChatMessage"] { background: #14171C; border: 1px solid #22272F; border-radius: 16px; padding: 0.8rem 1rem; }
[data-testid="stChatMessageAvatarCustom"] img, [data-testid="stChatMessage"] img { image-rendering: pixelated; border-radius: 10px; }
[data-testid="stExpander"] details, [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px !important; border-color: #22272F !important; }
.stButton button { border-radius: 999px; font-weight: 600; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background: #E5322D; border-color: #E5322D; color: #fff; }
[data-testid="stAudio"] { display: none; }
video { image-rendering: pixelated; border-radius: 14px; border: 1px solid #22272F; }
.runway { height: 6px; margin: 2px 0 14px 0; border-radius: 3px;
  background: repeating-linear-gradient(90deg, #C6F432 0 26px, transparent 26px 46px); opacity: .85; }
.ops-title { font-family: 'VT323', monospace; font-size: 2.6rem; color: #C6F432; letter-spacing: .06em; line-height: 1; }
.ops-sub { color: #9AA3AE; font-size: .9rem; margin-top: .3rem; }
.fids { background: #050607; border: 1px solid #262B33; border-radius: 14px; padding: 10px 14px; margin: 6px 0 12px 0;
  font-family: 'VT323', monospace; font-size: 1.35rem; color: #FFC233; overflow-x: auto; }
.fids table { width: 100%; border-collapse: collapse; }
.fids th { color: #6F7782; font-size: 1rem; text-align: left; font-weight: 400; padding: 0 10px 4px 0; letter-spacing: .08em; }
.fids td { padding: 2px 10px 2px 0; white-space: nowrap; }
.fids .ok { color: #C6F432; } .fids .bad { color: #FF5A4E; } .fids .busy { animation: blink 1.2s steps(2, start) infinite; }
@keyframes blink { to { visibility: hidden; } }
</style>
"""


def fire_test_alert() -> list[str]:
    """Run the whole notification chain on demand: toast + sound here, and every external channel."""
    from escalation import push
    sent = push("🧪 TEST ALARM · WSSS: Thunderstorm at the field",
                "Test alert from A-CDM Ops. If you can see/hear this, notifications work.", "ALARM")
    st.toast("🧪 TEST ALARM · Thunderstorm at the field", icon="🚨")
    if not play_alert():
        st.toast("Sound is off: turn on 🔔 Alert sounds in the sidebar", icon="🔕")
    return sent


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


TITLE = "AIR CONTROL MANAGEMENT CENTER"


def hero(subtitle: str = "", compact: bool = False) -> None:
    left, right = st.columns([1, 2] if not compact else [1, 4], vertical_alignment="center")
    video = ROOT / "assets" / "roster.mp4"
    if video.exists():
        left.video(str(video), autoplay=True, loop=True, muted=True)
    right.markdown(f"<div class='ops-title'>{TITLE}</div>" + (f"<div class='ops-sub'>{subtitle}</div>" if subtitle else ""),
                   unsafe_allow_html=True)
    st.markdown("<div class='runway'></div>", unsafe_allow_html=True)


def fids_board(s: Scenario, outcome=None, running: bool = False, pending: bool = False, emergency=None) -> None:
    """Departure-board style status strip for the active flight."""
    p = outcome.proposal if outcome else None
    if emergency is not None:
        p = emergency.proposal or Proposal(landing_slot_min=s.eta_min, runway="DIVERT", gate=emergency.divert_to or "ALTN")
        status, cls = f"{KIND_LABEL[emergency.kind]}", "bad busy"
    elif outcome is None:
        status, cls = (("AWAITING OPS", "busy") if pending else ("NEGOTIATING", "busy") if running
                       else (f"DELAYED +{s.late_aircraft_delay_min + s.weather_delay_min}", ""))
    elif p.runway == "DIVERT":
        status, cls = f"DIVERTED {p.gate}", "bad"
    else:
        status, cls = {"CONSENSUS": ("PLAN AGREED", "ok"), "FALLBACK": ("SOP PLAN", "bad"),
                       "HUMAN_DECISION": ("OPS DECISION", "ok")}[outcome.status]
    tldt = s.clock(p.landing_slot_min) if p is not None and p.runway != "DIVERT" else "--:--"
    rwy = p.runway if p is not None and p.runway != "DIVERT" else "---"
    stand = p.gate if p is not None and p.runway != "DIVERT" else "---"
    st.markdown(
        "<div class='fids'><table><tr><th>FLIGHT</th><th>FROM</th><th>TO</th><th>SIBT</th><th>ELDT</th><th>TLDT</th>"
        "<th>RWY</th><th>STAND</th><th>REMARKS</th></tr>"
        f"<tr><td>{s.flight}</td><td>{s.origin}</td><td>{s.destination}</td><td>{s.clock(0)}</td>"
        f"<td>{s.clock(s.eta_min)}</td><td>{tldt}</td><td>{rwy}</td><td>{stand}</td>"
        f"<td class='{cls}'>{status}</td></tr></table></div>", unsafe_allow_html=True)


@st.dialog("🎨 Customise your agents", width="large")
def customise_dialog() -> None:
    p = profiles.load()
    tabs = st.tabs([p[a]["name"] for a in ("MODERATOR", "ATC", "PILOT", "GROUND")])
    for tab, agent in zip(tabs, ("MODERATOR", "ATC", "PILOT", "GROUND")):
        with tab:
            c1, c2 = st.columns([1, 3])
            av = profiles.avatar_for_streamlit(agent, p)
            if av.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                c1.image(av, width=96)
            else:
                c1.markdown(f"<div style='font-size:64px'>{av}</div>", unsafe_allow_html=True)
            name = c2.text_input("Display name", p[agent]["name"], key=f"cz_name_{agent}")
            emoji = c2.text_input("Emoji avatar (optional)", "", key=f"cz_emoji_{agent}", placeholder="e.g. 🛫")
            up = c2.file_uploader("…or upload an image / GIF", type=["png", "jpg", "jpeg", "gif", "webp"],
                                  key=f"cz_up_{agent}")
            b1, b2 = c2.columns(2)
            if b1.button("Save", key=f"cz_save_{agent}", type="primary"):
                profiles.save(agent, name=name, avatar=emoji.strip() or None,
                              image_bytes=up.getvalue() if up else None,
                              ext=(up.name.rsplit(".", 1)[-1].lower() if up else "png"))
                st.rerun()
            if b2.button("Reset to default", key=f"cz_reset_{agent}"):
                profiles.reset(agent)
                st.rerun()


@st.cache_data(show_spinner="Loading DOT flights…")
def flights(limit: int = 3000) -> pd.DataFrame:
    df = load_disrupted_flights(limit=limit)
    df["SPLIT"] = [split_of(r) for _, r in df.iterrows()]
    return df


@st.cache_resource(show_spinner="Building scenario: DOT traffic, runways, weather sources…", max_entries=32)
def get_scenario(idx: int, weather_mode: str) -> Scenario:
    return build_scenario(load_disrupted_flights(limit=None).loc[idx], weather_mode=weather_mode)


@st.cache_data
def chime(repeats: int = 2) -> bytes:
    """Soft two-tone alert chime (880 Hz then 660 Hz, smooth attack/release), generated so no audio file is needed."""
    rate, buf = 22050, io.BytesIO()
    frames = bytearray()
    for _ in range(repeats):
        for freq in (880, 660):
            n = int(rate * 0.16)
            for i in range(n):
                env = min(1.0, i / (rate * 0.01), (n - i) / (rate * 0.04))  # 10 ms attack, 40 ms release
                frames += struct.pack("<h", int(9000 * env * math.sin(2 * math.pi * freq * i / rate)))
            frames += b"\x00\x00" * int(rate * 0.06)
        frames += b"\x00\x00" * int(rate * 0.25)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


beep = chime  # backwards-compatible name


def sounds_on() -> bool:
    return bool(st.session_state.get("sound_enabled"))


def play_alert(force: bool = False) -> bool:
    """Play the chime if the user has turned on 🔔 Alert sounds (that click unlocks browser audio).
    The player is hidden by CSS. Returns whether a sound was played."""
    if not (force or sounds_on()):
        return False
    st.audio(chime(), format="audio/wav", autoplay=True)
    return True


VERB = {"PROPOSE": "proposes", "ACCEPT": "agrees", "COUNTER": "suggests instead"}


def plan_text(s: Scenario, p) -> str:
    return f"land **{s.clock(p.landing_slot_min)}** · runway **{p.runway}** · stand **{p.gate}**"


def render_event(s: Scenario, kind: str, data, live: bool = False, sound: bool = True, autonomy: str = "") -> None:
    """Render one negotiation event. Chat-style events open their own chat bubble."""
    if kind == "turn":
        st.markdown(f"##### Round {data['round']} · turn {data['round_turn']} of {MAX_TURNS}")
    elif kind in ("ratification", "ratification_request"):
        with st.chat_message("MODERATOR", avatar=AVATARS["MODERATOR"]):
            st.markdown(f"**{NAMES['MODERATOR']}** · Plan #{data.plan_id} is agreed and needs all three stakeholders "
                        f"to ratify it: {plan_text(s, data.proposal)}")
            if data.chief_summary:
                st.write(data.chief_summary)
            if data.chief_reasoning:
                st.markdown(f"*My reasoning:* {data.chief_reasoning}")
            for fact in data.facts:
                st.markdown(f"- {fact}")
            for risk in data.risks:
                st.markdown(f"- ⚠️ {risk}")
            for p in data.positions:
                st.markdown(f"**{NAMES[p.agent]}** {VERB.get(p.stance, p.stance.lower())}: {p.rationale}")
            if any(p.constraints for p in data.positions):
                with st.expander("🔧 Technical details"):
                    for p in data.positions:
                        st.caption(f"{NAMES[p.agent]}: " + "; ".join(p.constraints) + f" · gives: {p.concession}")
    elif kind == "vote":
        with st.chat_message(data.agent, avatar=AVATARS[data.agent]):
            who = "person" if data.by == "HUMAN" else "AI agent"
            st.markdown(f"**{NAMES[data.agent]}** ({who}) "
                        + (":green-badge[AGREE]" if data.agree else ":red-badge[DISAGREE]") + f" {data.reason}")
    elif kind == "message":
        with st.chat_message(data.agent, avatar=AVATARS[data.agent]):
            tag = f"plan {data.plan_id}" if data.stance == "PROPOSE" else f"on plan {data.answering_plan}"
            st.markdown(f"**{NAMES[data.agent]}** :{STANCE[data.stance]}-badge[{VERB[data.stance]}] "
                        f"{plan_text(s, data.proposal)} :gray[({tag})]")
            if data.thinking:
                with st.expander("💭 Thinking"):
                    st.caption(data.thinking)
            st.write(data.rationale)
            st.caption(f"Willing to give: {data.concession}")
            if data.constraints:
                with st.expander(f"🔧 Technical details ({len(data.constraints)} limits published)"):
                    for c in data.constraints:
                        st.markdown(f"{'🔒' if c.hard else '◦'} **{c.name}**: {c.value}")
            meta = ""
            if data.metric:
                meta += (f" · {data.metric.latency_s}s · {data.metric.input_tokens}+{data.metric.output_tokens} tok"
                         f" · ${data.metric.cost_usd:.4f}")
            if meta:
                st.caption(meta.lstrip(" ·"))
    elif kind == "brief":
        with st.chat_message("MODERATOR", avatar=AVATARS["MODERATOR"]):
            st.markdown(f"**{NAMES['MODERATOR']}** · {data.summary}")
            if data.thinking:
                with st.expander("💭 Thinking"):
                    st.caption(data.thinking)
            for c in data.open_conflicts:
                st.markdown(f"- ⚠️ {c}")
            if data.bridge_proposal:
                st.markdown("Bridge plan: " + plan_text(s, data.bridge_proposal))
            for d in data.directives:
                st.caption(f"→ **{NAMES[d.agent]}**: {d.instruction}")
    elif kind == "system":
        icon = {"consensus": "✅", "ratified": "✅", "fallback": "🛑", "stale_reply": "🕓", "human_rejected": "👤",
                "stakeholder_objection": "↩️", "moderator_unavailable": "⚠️", "operator_directive": "🧭",
                "firewall_refused": "🛡️", "negotiation_stopped": "⏸️", "negotiation_restarted": "↻"}.get(data["kind"], "🚫")
        with st.chat_message("SYSTEM", avatar=AVATARS["SYSTEM"]):
            st.markdown(f"{icon} **{data['kind'].replace('_', ' ').upper()}**: {data['text']}")
    elif kind == "validation":
        rows = [{"domain": d, "check": c.name, "type": "HARD" if c.hard else "soft", "ok": "✅" if c.passed else "❌",
                 "detail": c.detail} for d, r in data["reports"].items() for c in r.checks]
        with st.expander(f"Deterministic validation of plan #{data['plan_id']}"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    elif kind == "alarm":
        st.error(f"🚨 **{data.severity} · {data.code}**: {data.title}. {data.detail}  \n"
                 f"Escalated via: {', '.join(data.channels)}")
        if live:
            st.toast(f"🚨 {data.title}", icon="🚨")
            if sound and data.severity == "ALARM":
                play_alert()
        if data.severity == "ALARM" and duty_phone():
            st.link_button(f"📞 Call duty manager {duty_phone()}", f"tel:{duty_phone()}")
    elif kind == "decision_auto":
        st.caption(f"🤖 Auto-decision ({autonomy.lower()}): {data['request'].title} → **{data['choice']}**")
    elif kind == "decision":
        st.caption(f"👤 Human decision requested: {data.title}")


def ratification_form(req, key: str) -> dict | None:
    """One panel per stakeholder: agree or disagree with a reason. Returns their votes once all three are complete."""
    answers = {}
    with st.container(border=True):
        st.markdown(f"#### 👥 Stakeholder votes · plan #{req.plan_id} · round {req.round}")
        st.caption("Any disagreement sends the plan back for a new round. If the next plan is rejected too, the "
                   "deterministic planner decides.")
        plan_key = f"{key}_{req.round}_{req.plan_id}"  # a new plan gets a blank card: no vote carries over
        with st.form(plan_key):
            for col, agent in zip(st.columns(3), STAKEHOLDERS):
                with col:
                    st.markdown(f"**{STAKEHOLDER_TITLE[agent]}**")
                    choice = st.radio("Vote", ["Agree", "Disagree"], index=None, horizontal=True,
                                      key=f"{plan_key}_{agent}_vote")
                    reason = st.text_area("Reason (required to disagree)", key=f"{plan_key}_{agent}_reason", height=90)
                    answers[agent] = (choice, reason.strip())
            send = st.form_submit_button("Send votes", type="primary")
    if not send:
        return None
    unanswered = [STAKEHOLDER_TITLE[a] for a, (choice, _) in answers.items() if choice is None]
    unexplained = [STAKEHOLDER_TITLE[a] for a, (choice, reason) in answers.items() if choice == "Disagree" and not reason]
    if unanswered or unexplained:
        if unanswered:
            st.warning(f"Still needs a vote: {', '.join(unanswered)}.")
        if unexplained:
            st.warning(f"Still needs a reason for disagreeing: {', '.join(unexplained)}.")
        return None
    return {a: {"agree": choice == "Agree", "reason": reason} for a, (choice, reason) in answers.items()}


def render_outcome(s: Scenario, out) -> None:
    p = out.proposal
    where = plan_text(s, p) if p.runway != "DIVERT" else f"**DIVERT to {p.gate}**"
    {"CONSENSUS": st.success, "FALLBACK": st.error, "HUMAN_DECISION": st.warning}[out.status](
        f"**{out.status.replace('_', ' ')}** · {out.plan_source} · round {out.rounds}, {out.turns_used} turn(s) in "
        f"total: {where}"
        + (f"\n\n{out.reason}" if out.reason else ""))
    c = st.columns(5)
    pm = out.reports["PILOT"].metrics if out.reports else {}
    gm = out.reports["GROUND"].metrics if out.reports else {}
    c[0].metric("Plan cost (USD)", f"${out.cost:,.0f}" if out.cost is not None else "–",
                None if out.cost is None or out.reference.cost is None
                else f"{out.cost - out.reference.cost:+,.0f} vs optimum", delta_color="inverse")
    c[1].metric("Holding / EFOB", f"{pm.get('holding_min', '–')} min", f"{pm.get('efob_kg', '–')} kg", delta_color="off")
    c[2].metric("Missed pax / overtime", f"{pm.get('missed_connection_pax', '–')} pax",
                f"{gm.get('crew_overtime_min', '–')} min", delta_color="off")
    c[3].metric("API spend", f"${out.usage['cost_usd']:.3f}", f"{out.usage['calls']} calls", delta_color="off")
    c[4].metric("Elapsed", f"{out.elapsed_s}s", f"{len(out.alarms)} alarms", delta_color="off")
    if out.reports and all(r.hard_ok for r in out.reports.values()):
        with st.expander("💲 Why this plan: cost breakdown (only safe plans are ever priced)"):
            items = {k: v for k, v in cost_breakdown(s, p, out.reports).items() if v}
            st.dataframe(pd.DataFrame({"item": list(items), "USD": [round(v) for v in items.values()]}),
                         hide_index=True, width="stretch")
            others = [out.reference.proposal] + [a for a, _ in out.reference.alternatives]
            rob = plan_robustness(s, p, [o for o in others if o != p])
            st.caption(f"Robustness: this plan stays the cheapest safe option in **{rob:.0%}** of 300 scenarios "
                       "where every weight is varied ±50%. Weights and sources: costs.py / Control room sidebar.")
    if out.log_path:
        st.download_button(f"Download consensus log (hash chain {'verified ✅' if out.log_verified else 'BROKEN ❌'})",
                           Path(out.log_path).read_text(), file_name=Path(out.log_path).name,
                           mime="application/jsonl", key=f"dl_{out.log_path}")


def timeline(s: Scenario, final=None, reference=None) -> go.Figure:
    fig = go.Figure()
    for r in s.runways:
        if r.closed_until > -999:
            fig.add_shape(type="line", x0=s.eta_min - 20, x1=r.closed_until, y0=r.id, y1=r.id,
                          line=dict(color="rgba(214,39,40,0.35)", width=18))
        for src, colour in (("DOT", "#4c78a8"), ("SIM", "#b8bec9")):
            pts = [t for t in r.traffic if t.source == src]
            fig.add_trace(go.Scatter(x=[t.t for t in pts], y=[r.id] * len(pts), mode="markers",
                                     name="DOT 2015 arrival" if src == "DOT" else "simulated traffic",
                                     legendgroup=src, showlegend=r is s.runways[0],
                                     marker=dict(size=[12 if t.wake in ("SUPER", "HEAVY") else 8 for t in pts], color=colour),
                                     text=[f"{t.callsign} {t.wake} {s.clock(t.t)}" for t in pts], hoverinfo="text"))
    fig.add_vline(x=s.eta_min, line_dash="dot", line_color="#4c78a8", annotation_text="ELDT")
    fig.add_vline(x=s.latest_safe_landing, line_dash="dash", line_color="#d62728", annotation_text="fuel limit")
    for p, name, colour, sym in [(reference, "deterministic optimum", "#9467bd", "diamond-open"),
                                 (final, "final plan", "#2ca02c", "star")]:
        if p is not None and s.runway(p.runway):
            fig.add_trace(go.Scatter(x=[p.landing_slot_min], y=[p.runway], mode="markers", name=name,
                                     marker=dict(size=20, color=colour, symbol=sym, line=dict(width=2))))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.4),
                      xaxis=dict(title="minutes relative to SIBT (red band = runway closed)",
                                 range=[s.eta_min - 20, max(s.latest_safe_landing, s.eta_min + 60) + 15]))
    return fig


def render_emergency(s: Scenario, plan: EmergencyPlan, readbacks: dict | None = None, live: bool = False,
                     sound: bool = True) -> None:
    head = (f"🚨 **{KIND_LABEL[plan.kind]} · {plan.flight}**"
            + (f" — {plan.reason}" if plan.reason else "")
            + f"  \nDecided in **{plan.decision_ms:.2f} ms** by the deterministic emergency planner (no negotiation). ")
    if plan.divert_to:
        st.error(head + f"  \n**DIVERT to {plan.divert_to}**: no usable runway before fuel runs out.")
    else:
        p = plan.proposal
        st.error(head + f"  \nPriority landing **RWY {p.runway} at {plan.landing_clock}** (no holding) · stand "
                 f"**{p.gate}** in-block {plan.in_block_clock} · EFOB {plan.efob_kg} kg ({plan.fuel_status})")
    if live:
        st.toast(f"🚨 {KIND_LABEL[plan.kind]} {plan.flight}", icon="🚨")
        if sound:
            play_alert()
    cols = st.columns(3)
    for col, key in zip(cols, ("ATC", "PILOT", "GROUND")):
        with col.container(border=True):
            av = AVATARS[key]
            h1, h2 = st.columns([1, 4], vertical_alignment="center")
            if str(av).endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                h1.image(av, width=36)
            else:
                h1.markdown(f"<div style='font-size:26px'>{av}</div>", unsafe_allow_html=True)
            h2.markdown(f"**{NAMES[key]}**")
            for a in plan.actions[key]:
                st.markdown(f"- {a}")
            if readbacks and readbacks.get(key):
                st.caption(f"📻 Readback: {readbacks[key]}")
    if duty_phone():
        st.link_button(f"📞 Call duty manager {duty_phone()}", f"tel:{duty_phone()}")


# ============================================================================
# Readable weather panel (replaces the one-line attribution banner)
# ============================================================================
def weather_card(s: Scenario) -> None:
    wx = s.weather
    if wx is None:
        st.info("Weather: no source available for this flight (treated as unknown).", icon="🌦️")
        return
    wind = (f"{wx.wind_dir_deg:03.0f}° / {wx.wind_kt:.0f} kt" + (f", gusts {wx.gust_kt:.0f}" if wx.gust_kt else "")
            if wx.wind_kt is not None and wx.wind_dir_deg is not None else "n/a")
    phenomena = [x for x, on in (("⚡ Thunderstorm", wx.thunderstorm), ("☁️ CB / convection", wx.convective_nearby
                                 and not wx.thunderstorm), ("❄️ Freezing precip / snow", wx.freezing_precip_or_snow)) if on]
    impact = ("🔴 Destination weather is affecting arrivals" if s.weather_at_destination
              else "🟢 Destination is fine: the WEATHER_DELAY happened en route / at the origin")
    with st.container(border=True):
        st.markdown(f"**🌦️ Weather at {wx.airport_icao}** · {wx.mode.lower()} · valid {wx.valid_local} local  \n{impact}")
        c = st.columns(5)
        c[0].metric("Conditions", wx.flight_category)
        c[1].metric("Wind", wind)
        c[2].metric("Visibility", f"{wx.visibility_sm:g} sm" if wx.visibility_sm is not None else "n/a")
        c[3].metric("Ceiling", f"{wx.ceiling_ft:,} ft" if wx.ceiling_ft else "none")
        c[4].metric("Holding PIREPs", f"turb {wx.hold_turbulence}", f"icing {wx.hold_icing} · {len(wx.pireps)} rpt",
                    delta_color="off")
        rows = [{"Runway": r.id, "Status": "Open" if r.closed_until <= -999 else f"Closed until {s.clock(r.closed_until)}",
                 "Reason": "" if r.closed_until <= -999 else r.closure_reason,
                 "Crosswind kt": r.crosswind_kt, "Tailwind kt": r.tailwind_kt} for r in s.runways]
        left, right = st.columns([3, 2])
        left.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        right.markdown("**Hazards**  \n" + ("  \n".join(f"- {p}" for p in phenomena) or "- none reported")
                       + (f"  \n- 🌩️ Ramp closed (lightning) until {s.clock(s.ramp_closed_until)}"
                          if s.ramp_closed_until > -999 else ""))
        if wx.metar:
            st.caption(f"METAR {wx.metar}")


# ============================================================================
# MAYDAY: available on every page, behind a confirmation
# ============================================================================
@st.dialog("🚨 Declare MAYDAY")
def mayday_dialog() -> None:
    from emergency import KIND_LABEL
    st.markdown("This **abandons any negotiation** and issues the deterministic emergency plan to every desk at once: "
                "priority landing, stand and fire/medical services. Alarms and notifications go out immediately.")
    kind = st.radio("Emergency type", list(KIND_LABEL), format_func=KIND_LABEL.get, horizontal=True)
    reason = st.text_input("Reason (optional)", placeholder="e.g. engine fire on approach")
    c1, c2 = st.columns(2)
    if c1.button("CONFIRM MAYDAY", type="primary", width="stretch"):
        st.session_state.mayday_request = {"kind": kind, "reason": reason}
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()
