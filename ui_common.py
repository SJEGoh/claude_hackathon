"""Rendering helpers shared by the Ops chat and the Control room pages."""
import io
import math
import struct
import wave
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from escalation import duty_phone
from negotiation import MAX_TURNS
from scenario import Scenario, build_scenario, load_disrupted_flights, split_of

ROOT = Path(__file__).parent
AVATARS = {"ATC": "🗼", "PILOT": "🧑‍✈️", "GROUND": "🧳", "MODERATOR": "🧭", "SYSTEM": "⚙️", "user": "👤"}
NAMES = {"ATC": "ATC · Arrival Manager", "PILOT": "Flight Crew · Captain", "GROUND": "AOCC · Turnaround",
         "MODERATOR": "Chief of Staff", "SYSTEM": "Safety engine"}
STANCE = {"PROPOSE": "blue", "ACCEPT": "green", "COUNTER": "orange"}


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
    c[0].metric("Plan cost", out.cost, None if out.cost is None or out.reference.cost is None
                else f"{out.cost - out.reference.cost:+.1f} vs optimum", delta_color="inverse")
    c[1].metric("Holding / EFOB", f"{pm.get('holding_min', '–')} min", f"{pm.get('efob_kg', '–')} kg", delta_color="off")
    c[2].metric("Missed pax / overtime", f"{pm.get('missed_connection_pax', '–')} pax",
                f"{gm.get('crew_overtime_min', '–')} min", delta_color="off")
    c[3].metric("API spend", f"${out.usage['cost_usd']:.3f}", f"{out.usage['calls']} calls", delta_color="off")
    c[4].metric("Elapsed", f"{out.elapsed_s}s", f"{len(out.alarms)} alarms", delta_color="off")
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
