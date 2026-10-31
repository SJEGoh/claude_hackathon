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
from negotiation import MAX_TURNS
from scenario import Scenario, build_scenario, load_disrupted_flights, split_of

ROOT = Path(__file__).parent
STANCE = {"PROPOSE": "blue", "ACCEPT": "green", "COUNTER": "orange"}
ENGINES = ["Claude agents", "Rule-based (free)", "Safety drill: rogue agents (free)"]


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
    if engine.startswith("Rule"):
        return make_rule_agents()
    if engine.startswith("Safety drill"):
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


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(subtitle: str, compact: bool = False) -> None:
    left, right = st.columns([1, 2] if not compact else [1, 4], vertical_alignment="center")
    video = ROOT / "assets" / "roster.mp4"
    if video.exists():
        left.video(str(video), autoplay=True, loop=True, muted=True)
    right.markdown(f"<div class='ops-title'>A-CDM OPS CENTER</div><div class='ops-sub'>{subtitle}</div>",
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


def beep() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(22050)
        for i in range(int(22050 * 0.9)):
            on = (i // 3300) % 2 == 0
            w.writeframes(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 880 * i / 22050)) if on else 0))
    return buf.getvalue()


def plan_text(s: Scenario, p) -> str:
    return f"TLDT **{s.clock(p.landing_slot_min)}** (slot {p.landing_slot_min}) · RWY **{p.runway}** · stand **{p.gate}**"


def render_event(s: Scenario, kind: str, data, live: bool = False, sound: bool = True, autonomy: str = "") -> None:
    """Render one negotiation event. Chat-style events open their own chat bubble."""
    if kind == "turn":
        st.markdown(f"##### — Turn {data['turn']} / {MAX_TURNS} —")
    elif kind == "message":
        with st.chat_message(data.agent, avatar=AVATARS[data.agent]):
            tag = f"plan #{data.plan_id}" if data.stance == "PROPOSE" else f"re #{data.answering_plan}"
            st.markdown(f"**{NAMES[data.agent]}** :{STANCE[data.stance]}-badge[{data.stance}] `{tag}` "
                        f"{plan_text(s, data.proposal)}")
            st.write(data.rationale)
            if data.constraints:
                with st.expander(f"{len(data.constraints)} published constraints"):
                    for c in data.constraints:
                        st.markdown(f"{'🔒' if c.hard else '◦'} **{c.name}**: {c.value}")
            meta = f"Concession: {data.concession}"
            if data.metric:
                meta += (f" · {data.metric.latency_s}s · {data.metric.input_tokens}+{data.metric.output_tokens} tok"
                         f" · ${data.metric.cost_usd:.4f}")
            st.caption(meta)
    elif kind == "brief":
        with st.chat_message("MODERATOR", avatar=AVATARS["MODERATOR"]):
            st.markdown(f"**{NAMES['MODERATOR']}** · {data.summary}")
            for c in data.open_conflicts:
                st.markdown(f"- ⚠️ {c}")
            if data.bridge_proposal:
                st.markdown("Bridge plan: " + plan_text(s, data.bridge_proposal))
            for d in data.directives:
                st.caption(f"→ {AVATARS[d.agent]} {d.instruction}")
    elif kind == "system":
        icon = {"consensus": "✅", "fallback": "🛑", "stale_reply": "🕓", "human_rejected": "👤"}.get(data["kind"], "🚫")
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
                st.audio(beep(), format="audio/wav", autoplay=True)
        if data.severity == "ALARM" and duty_phone():
            st.link_button(f"📞 Call duty manager {duty_phone()}", f"tel:{duty_phone()}")
    elif kind == "decision_auto":
        st.caption(f"🤖 Auto-decision ({autonomy.lower()}): {data['request'].title} → **{data['choice']}**")
    elif kind == "decision":
        st.caption(f"👤 Human decision requested: {data.title}")


def render_outcome(s: Scenario, out) -> None:
    p = out.proposal
    where = plan_text(s, p) if p.runway != "DIVERT" else f"**DIVERT to {p.gate}**"
    {"CONSENSUS": st.success, "FALLBACK": st.error, "HUMAN_DECISION": st.warning}[out.status](
        f"**{out.status.replace('_', ' ')}** · {out.plan_source} · turn {out.turns_used}: {where}"
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
                                     marker=dict(size=[12 if t.wake == "HEAVY" else 8 for t in pts], color=colour),
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
            st.audio(beep(), format="audio/wav", autoplay=True)
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
