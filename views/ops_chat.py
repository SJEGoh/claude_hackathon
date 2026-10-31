
"""Ops chat: a team chat (#ops-team) plus a private DM with every agent, like a Grok-style bot roster.

- #ops-team: you, the Chief of Staff and the three specialists. Messages go to the Chief unless you
  @mention someone (@ATC, @Pilot, @Ground, @Chief). The bounded negotiation plays out here.
- DMs: private 1:1 chats. Each agent keeps its own memory per channel; other agents never see your DMs.
"""
import json
import os
import re
from datetime import datetime

import anthropic
import streamlit as st

import config  # noqa: F401  (loads .env)
from agents import DEFAULT_MODEL, ROLES, render_brief, render_message
from agents.base import SAFETY_CHARTER
from agents.chief import CHIEF_SYSTEM, CHIEF_TOOLS
from agents.conversation import CHAT_STYLE, Conversation, converse, specialist_tools
from monitoring import summarise
from constraints import Proposal, check_all, cost, cost_breakdown, is_safe
from emergency import KIND_LABEL, plan_emergency, readbacks
from escalation import Alarm, dispatch
from negotiation import Autonomy, ConsensusLog, NegotiationSession
from scenario import load_disrupted_flights
import voice
import steer
from agents import make_agents
from baseline_agents import make_rule_agents
from ui_common import (AVATARS, ENGINES, NAMES, ROOT, fids_board, get_scenario, hero, make_engine_agents, weather_card,
                       ratification_form, render_emergency, render_event, render_outcome, timeline)

CHANNELS = ["group", "MODERATOR", "ATC", "PILOT", "GROUND"]
LABEL = {"group": "👥 #ops-team", "MODERATOR": "✉️ Chief of Staff", "ATC": "✉️ ATC", "PILOT": "✉️ Flight Crew",
         "GROUND": "✉️ AOCC"}  # fixed labels: Streamlit loses a widget whose labels change
STATUS_ICON = {"idle": "🟢", "working": "🟡", "waiting": "🔴"}
MENTIONS = {"@atc": "ATC", "@pilot": "PILOT", "@crew": "PILOT", "@captain": "PILOT", "@ground": "GROUND",
            "@aocc": "GROUND", "@chief": "MODERATOR", "@cos": "MODERATOR"}
CHAT_LOG = ROOT / "logs" / f"chat_{datetime.now():%Y%m%d}.jsonl"

S = st.session_state
S.setdefault("oc_items", {c: [] for c in CHANNELS})
S.setdefault("oc_convs", {})
S.setdefault("oc_status", {a: "idle" for a in ROLES})
S.setdefault("oc_unread", {c: 0 for c in CHANNELS})
S.setdefault("oc_group_log", [])
S.setdefault("oc_scenario", None)
S.setdefault("oc_session", None)
S.setdefault("oc_pending", None)
S.setdefault("oc_ratify", None)
S.setdefault("oc_votes", None)
S.setdefault("oc_outcome", None)
S.setdefault("oc_metrics", [])
S.setdefault("oc_click", None)
S.setdefault("oc_channel", "group")
S.setdefault("oc_emergency", None)
S.setdefault("oc_fire", None)

# ============================================================================
# Sidebar: channels (with each agent's picture and status) + settings
# ============================================================================
CHANNEL_NAME = {"group": "#ops-team", "MODERATOR": "Chief of Staff", "ATC": "ATC", "PILOT": "Flight Crew",
                "GROUND": "AOCC"}
DOT = {"online": "🟢", "busy": "🔴", "offline": "🟡"}
KEY_OK = len(os.environ.get("ANTHROPIC_API_KEY", "")) > 40

with st.sidebar:
    st.subheader("Channels")
    dots: dict[str, "st.delta_generator.DeltaGenerator"] = {}
    for c in CHANNELS:
        a1, a2, a3 = st.columns([1, 4, 1.2], vertical_alignment="center")
        av = "👥" if c == "group" else AVATARS[c]
        if str(av).endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            a1.image(av, width=30)
        else:
            a1.markdown(f"<div style='font-size:22px;text-align:center'>{av}</div>", unsafe_allow_html=True)
        a2.button(CHANNEL_NAME[c], key=f"ch_{c}", width="stretch",
                  type="primary" if c == S.oc_channel else "secondary",
                  on_click=lambda c=c: S.update(oc_channel=c))
        dots[c] = a3.empty()

    def availability(agent: str) -> str:
        if not KEY_OK or S.oc_status.get(agent) == "error":
            return "offline"
        return "busy" if S.oc_status.get(agent) in ("working", "waiting") else "online"

    def draw_dot(c: str) -> None:
        unread = S.oc_unread.get(c, 0) if c != S.oc_channel else 0
        badge = f"<span style='font-size:11px;opacity:.8'> {unread}</span>" if unread else ""
        mark = "" if c == "group" else DOT[availability(c)]
        dots[c].markdown(f"<div style='text-align:center' title='{availability(c) if c != 'group' else ''}'>"
                         f"{mark}{badge}</div>", unsafe_allow_html=True)

    def show_status(agent: str, status: str) -> None:
        S.oc_status[agent] = status
        if agent in dots:
            draw_dot(agent)

    for c in CHANNELS:
        draw_dot(c)
    st.caption("🟢 online · 🔴 busy · 🟡 offline")
    st.divider()
    with st.expander("Settings", expanded=False):
        st.caption("✅ API key loaded from .env" if KEY_OK else "⚠️ No API key: add it to .env")
        model = st.selectbox("Model", [DEFAULT_MODEL, "claude-sonnet-5-5", "claude-haiku-5-5"])
        effort = st.select_slider("Effort", ["low", "medium", "high"], value="medium", help="medium shows the agents' thinking summaries; low is cheaper and often skips thinking")
        engine = st.radio("Negotiation engine", ENGINES)
        default_autonomy = st.radio("Default autonomy", [a.value for a in Autonomy], index=1,
                                    format_func=lambda a: a.replace("_", " ").title())
        sound = st.toggle("Audible alarms", value=True)
    u = summarise(S.oc_metrics)
    st.caption(f"Chat spend this session: ${u['cost_usd']:.3f} · {u['calls']} calls")
    if st.button("🗑️ Clear all chats"):
        for k in list(S.keys()):
            if k.startswith("oc_"):
                del S[k]
        st.rerun()

channel = S.oc_channel
S.oc_unread[channel] = 0
client = anthropic.Anthropic(max_retries=1)


# ============================================================================
# Posting + rendering
# ============================================================================
def scenario():
    return S.oc_scenario


def draw(item: dict) -> None:
    t = item["t"]
    s = scenario()
    if t == "msg":
        who = item["who"]
        with st.chat_message("user" if who == "user" else who, avatar=AVATARS.get(who, "👤")):
            if who != "user":
                st.markdown(f"**{NAMES.get(who, who)}**" + (f" · _to the team_" if item.get("via") else ""))
            if item.get("thinking"):
                with st.expander("💭 Thinking"):
                    st.caption(item["thinking"])
            st.markdown(item["text"])
            if item.get("meta"):
                st.caption(item["meta"])
    elif t == "tool":
        st.caption(f"🔧 {NAMES.get(item['who'], item['who'])} used `{item['name']}"
                   f"({', '.join(f'{k}={v}' for k, v in item['args'].items())})`")
    elif t == "event" and s is not None:
        render_event(s, item["kind"], item["data"], live=item.get("live", False), sound=sound,
                     autonomy=S.oc_session.autonomy.value if S.oc_session else "")
        item["live"] = False
    elif t == "outcome" and s is not None:
        render_outcome(s, item["data"])
    elif t == "emergency" and s is not None:
        render_emergency(s, item["data"], item.get("rb"), live=item.get("live", False), sound=sound)
        item["live"] = False
    elif t == "note":
        st.caption(item["text"])
        if item.get("jump") and item["jump"] != channel:
            st.button(f"Open {LABEL[item['jump']]} →", key=f"jump_{id(item)}",
                      on_click=lambda target=item["jump"]: S.update(oc_channel=target))


def post(ch: str, item: dict, log_line: str | None = None) -> None:
    """Append to a channel; draw immediately if the channel is on screen."""
    S.oc_items[ch].append(item)
    if ch == channel:
        draw(item)
    else:
        S.oc_unread[ch] += 1
    if ch == "group" and log_line:
        S.oc_group_log.append(log_line)
    if item["t"] == "msg":
        CHAT_LOG.parent.mkdir(exist_ok=True)
        with CHAT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "channel": ch,
                                "who": item["who"], "text": item["text"]}) + "\n")


# ============================================================================
# Introductions: every channel opens with its agent(s) saying who they are
# ============================================================================
INTROS = {
    "MODERATOR": ("Hi, I'm **{name}**, your Duty Operations Manager. I coordinate the team below: I find and load "
                  "disrupted flights from the US DOT 2015 data, run the 3-turn negotiation between ATC, Flight Crew "
                  "and AOCC, and bring you any decision that needs a human. I have no private instruments, so for "
                  "specialist figures I ask the specialist. Try: *\"find a weather-delayed flight into ORD\"*."),
    "ATC": ("**{name}** here, approach and ground movement. I own the landing sequence: **runway wake "
            "separation**, runway closures and crosswind/tailwind limits, **taxiway blockages**, and **maximising "
            "arrival throughput**. I can test any landing slot against separation and list conflict-free AMAN "
            "slots. Fuel and passengers are the Captain's; stands and crew are AOCC's, and I'll hand those over."),
    "PILOT": ("**{name}**, flight deck. I **preserve fuel reserves** (never below final reserve plus alternate and "
              "contingency), stay inside crew duty limits, keep the flight safe, **minimise time in turbulence and icing** using PIREPs, and **protect "
              "passengers' connections**. I can give you FMS fuel predictions for any landing time and check a plan "
              "against my limits. Runway sequencing is ATC's; stands are AOCC's."),
    "GROUND": ("**{name}**, Airport Operations Control Centre. I **optimise stand utilisation**, keep **ground crew "
               "out of overtime**, give proper gate-change notice, and **minimise turnaround delay** and knock-on to "
               "the next departure. I can check any plan against stand, crew and turnaround limits. Landing times "
               "are ATC's; fuel is the Captain's."),
}


def intro(agent: str) -> dict:
    return {"t": "msg", "who": agent, "text": INTROS[agent].format(name=NAMES[agent]), "intro": True}


for _ch in CHANNELS:
    if not S.oc_items[_ch]:
        S.oc_items[_ch] = ([intro(a) for a in ("MODERATOR", "ATC", "PILOT", "GROUND")] if _ch == "group"
                           else [intro(_ch)])


# ============================================================================
# Agents in chat
# ============================================================================
AGENT_OBJECTS = {k: cls(None) for k, cls in ROLES.items()}  # for role text and private briefs only


def conv(agent: str, ch: str) -> Conversation:
    key = f"{agent}|{ch}"
    if key not in S.oc_convs:
        S.oc_convs[key] = Conversation(agent, ch)
    return S.oc_convs[key]


def system_for(agent: str, ch: str) -> str:
    where = ("This is #ops-team, the group chat: the duty officer, the Chief of Staff and all specialists read "
             "it." if ch == "group" else "This is a PRIVATE direct message with the duty officer; no other agent "
             "sees it.")
    if agent == "MODERATOR":
        return f"{CHIEF_SYSTEM}\n\n{where}"
    return f"{AGENT_OBJECTS[agent].role}\n\n{SAFETY_CHARTER}\n\n{CHAT_STYLE}\n\n{where}"


def context_for(agent: str, c: Conversation) -> str:
    parts, s = [], scenario()
    if not c.history:
        parts.append(f"## You opened this channel by introducing yourself as:\n{intro(agent)['text']}")
    sig = f"{s.flight}|{s.date}|{s.weather_mode}" if s else "none"
    if sig != c.context_sig:
        c.context_sig = sig
        if s is None:
            parts.append("## Active flight\nNone loaded yet.")
        elif agent == "MODERATOR":
            parts.append(f"## Active flight (public facts)\n{AGENT_OBJECTS['MODERATOR'].public_brief(s)}")
        else:
            parts.append(f"## Your private instruments for the active flight\n{AGENT_OBJECTS[agent].private_brief(s)}")
    new = S.oc_group_log[c.group_cursor:]
    c.group_cursor = len(S.oc_group_log)
    if new and c.channel != "group":
        parts.append("## Recent #ops-team activity (for awareness)\n" + "\n".join(new[-30:]))
    elif new:
        parts.append("## New in #ops-team since your last message\n" + "\n".join(new[-40:]))
    if agent == "MODERATOR" and S.oc_ratify is not None:
        parts.append(f"## WAITING FOR STAKEHOLDER VOTES on plan #{S.oc_ratify.plan_id}: the controller, pilot and ground "
                     "lead vote on the card in #ops-team. You cannot vote for them.")
    if agent == "MODERATOR" and S.oc_pending is not None:
        req = S.oc_pending
        parts.append(f"## PENDING DECISION ({req.code}): {req.title}\n" + "\n".join(req.briefing) + "\nOptions: "
                     + "; ".join(f"{o.id} = {o.label} ({o.description})" for o in req.options))
    return "\n\n".join(parts)


def make_refer(from_agent: str, ch: str):
    """Handoff: from_agent passes a question to the specialist who owns it. One hop only."""
    def refer(agent: str, question: str) -> str:
        if agent == from_agent:
            return "That is your own domain: answer it yourself."
        target_ch = "group" if ch == "group" else agent  # in a DM, the owner answers in their own DM
        post(ch, {"t": "note", "text": f"↪️ **{NAMES[from_agent]}** handed this to **{NAMES[agent]}**: _{question}_",
                  "jump": None if target_ch == ch else target_ch},
             log_line=f"{from_agent} referred to {agent}: {question}")
        if target_ch != ch:
            post(target_ch, {"t": "note", "text": f"↪️ Handed over by **{NAMES[from_agent]}** on your behalf."})
        answer = talk(agent, target_ch, question, allow_refer=False,
                      asked_by=f"{NAMES[from_agent]}, on behalf of the duty officer (handoff: answer the duty officer)")
        if target_ch != ch:  # mirror the answer where the duty officer asked
            post(ch, {"t": "msg", "who": agent, "text": answer, "meta": f"answered in {LABEL[target_ch]}"})
        return f"{NAMES[agent]} answered the duty officer: {answer}"
    return refer


def talk(agent: str, ch: str, text: str, asked_by: str = "the duty officer", allow_refer: bool = True) -> str:
    """One chat turn with an agent in a channel; posts its reply there and returns it."""
    c = conv(agent, ch)
    ctx = context_for(agent, c)
    content = (ctx + "\n\n" if ctx else "") + f"## Message from {asked_by}\n{text}"
    if agent == "MODERATOR":
        tools, impl = CHIEF_TOOLS, CHIEF_IMPL
    else:
        tools, impl = specialist_tools(agent, scenario, make_refer(agent, ch) if allow_refer else None)

    def on_tool(name, args, result):
        if name not in ("ask_specialist", "refer_to_specialist"):  # those post their own messages
            post(ch, {"t": "tool", "who": agent, "name": name, "args": args})

    show_status(agent, "working")
    with st.spinner(f"{NAMES[agent]} is typing…"):
        reply, metrics, thinking = converse(client, model, effort, system_for(agent, ch), c, content, tools, impl, on_tool)
    show_status(agent, "error" if metrics and not metrics[-1].ok else
                "waiting" if (S.oc_pending and agent == "MODERATOR") else "idle")
    S.oc_metrics.extend(metrics)
    cost = sum(m.cost_usd for m in metrics)
    post(ch, {"t": "msg", "who": agent, "text": reply, "thinking": thinking,
              "meta": f"{sum(m.latency_s for m in metrics):.1f}s · ${cost:.4f}"},
         log_line=f"{agent}: {reply}")
    return reply


# ============================================================================
# Chief of Staff tools (they drive the shared session)
# ============================================================================
def t_find_flights(destination_airport: str, bucket: str, min_weather_delay: int, limit: int) -> str:
    df = load_disrupted_flights(limit=None)
    if destination_airport:
        df = df[df.DESTINATION_AIRPORT == destination_airport.upper()]
    if bucket != "any":
        df = df[df.BUCKET == bucket]
    if min_weather_delay:
        df = df[df.WEATHER_DELAY >= min_weather_delay]
    rows = [{"row_id": int(i), "flight": f"{r.AIRLINE}{r.FLIGHT_NUMBER}", "route": f"{r.ORIGIN_AIRPORT}-{r.DESTINATION_AIRPORT}",
             "date": f"2015-{r.MONTH:02d}-{r.DAY:02d}", "sched_arr": f"{int(r.SCHEDULED_ARRIVAL):04d}",
             "weather_delay": int(r.WEATHER_DELAY), "late_aircraft_delay": int(r.LATE_AIRCRAFT_DELAY),
             "bucket": r.BUCKET} for i, r in df.head(max(1, min(limit, 10))).iterrows()]
    return json.dumps({"matches": len(df), "showing": rows})


def t_load_flight(row_id: int, weather_mode: str) -> str:
    s = get_scenario(row_id, weather_mode)
    S.oc_scenario, S.oc_session, S.oc_pending, S.oc_outcome, S.oc_emergency = s, None, None, None, None
    S.oc_ratify = None
    post("group", {"t": "note", "text": f"📂 Active flight: **{s.flight}** {s.origin}→{s.destination} {s.date}, "
                                        f"SIBT {s.clock(0)} · WX {s.weather_delay_min} · LATE {s.late_aircraft_delay_min} "
                                        f"· weather {weather_mode.lower()}"},
         log_line=f"SYSTEM: active flight is now {s.flight} {s.origin}-{s.destination} {s.date}")
    return AGENT_OBJECTS["MODERATOR"].public_brief(s)


def t_flight_status() -> str:
    s = scenario()
    if s is None:
        return "No flight loaded."
    state = (f"waiting for stakeholder votes on plan #{S.oc_ratify.plan_id}" if S.oc_ratify else
             "paused for a decision: " + S.oc_pending.title if S.oc_pending else
             f"finished: {S.oc_outcome.status}" if S.oc_outcome else
             "running" if S.oc_session else "not started")
    return AGENT_OBJECTS["MODERATOR"].public_brief(s) + f"\nNegotiation: {state}."


def pump(choice=None) -> str:
    sess: NegotiationSession = S.oc_session
    s = scenario()
    note = None if channel == "group" else st.empty()
    S.oc_running, S.oc_halted = True, False
    ev = sess.advance(choice)
    while True:
        kind, data = ev
        if kind == "thinking":
            for a in data["agents"]:
                show_status(a, "working")
            if note:
                note.info(f"Negotiation running in #ops-team: {', '.join(data['agents'])} working…")
        elif kind != "metric":
            line = {"message": lambda: render_message(data, s), "brief": lambda: render_brief(data),
                    "system": lambda: f"SYSTEM {data['kind']}: {data['text']}",
                    "alarm": lambda: f"ALARM {data.code}: {data.title}",
                    "ratification": lambda: f"CHIEF OF STAFF to stakeholders: ratify plan #{data.plan_id}, {data.plan_text}",
                    "ratification_request": lambda: f"CHIEF OF STAFF to stakeholders: ratify plan #{data.plan_id}, {data.plan_text}",
                    "vote": lambda: f"{data.agent} ({data.by}) {'AGREE' if data.agree else 'DISAGREE'}: {data.reason}",
                    }.get(kind, lambda: None)()
            if kind == "message":
                show_status(data.agent, "idle")
            if kind == "brief":
                show_status("MODERATOR", "idle")
            post("group", {"t": "event", "kind": kind, "data": data, "live": True}, log_line=line)
        if kind in ("ratification", "decision", "outcome"):
            S.oc_running = False
        if kind == "ratification":
            S.oc_ratify = data
            for a in ("ATC", "PILOT", "GROUND"):
                show_status(a, "idle")
            show_status("MODERATOR", "waiting")
            return (f"PAUSED for stakeholder votes on plan #{data.plan_id}: {data.plan_text}. The controller, pilot and "
                    "ground lead vote on the card in #ops-team; you cannot vote for them. Tell the duty officer so.")
        if kind == "decision":
            S.oc_pending = data
            for a in ("ATC", "PILOT", "GROUND"):
                show_status(a, "idle")
            show_status("MODERATOR", "waiting")
            return (f"PAUSED for the duty officer: {data.title}. " + " ".join(data.briefing) + " Options: "
                    + "; ".join(f"{o.id} = {o.label}: {o.description}" for o in data.options))
        if kind == "outcome":
            S.oc_pending, S.oc_ratify, S.oc_outcome = None, None, data
            post("group", {"t": "outcome", "data": data},
                 log_line=f"OUTCOME {data.status} ({data.plan_source}) turn {data.turns_used}: "
                          f"{render_plan(data.proposal)}")
            for a in ROLES:
                show_status(a, "idle")
            return (f"OUTCOME {data.status} via {data.plan_source} at turn {data.turns_used}: "
                    f"{render_plan(data.proposal)}; cost {data.cost} vs deterministic optimum {data.reference.cost}; "
                    f"{len(data.alarms)} alarms; log hash chain {'verified' if data.log_verified else 'BROKEN'}.")
        ev = sess.advance()


def render_plan(p) -> str:
    s = scenario()
    return f"TLDT {s.clock(p.landing_slot_min)} (slot {p.landing_slot_min}) RWY {p.runway} stand {p.gate}"


def t_start_negotiation(autonomy: str) -> str:
    s = scenario()
    if s is None:
        return "No flight loaded. Use find_flights and load_flight first."
    if S.oc_pending is not None or S.oc_ratify is not None:
        return "A negotiation is already paused for a decision or stakeholder votes; finish that first."
    eng = engine
    agents = make_engine_agents(eng, model, effort)
    S.oc_session = NegotiationSession(s, agents, Autonomy(autonomy))
    S.oc_outcome = None
    post("group", {"t": "note", "text": f"▶ Negotiation started for {s.flight} · autonomy {autonomy.lower()} · "
                                        f"{model if eng == ENGINES[0] else eng.lower()}"},
         log_line=f"SYSTEM: negotiation started ({autonomy})")
    return pump()


def t_resolve_decision(option_id: str) -> str:
    req = S.oc_pending
    if req is None:
        return "There is no pending decision."
    if option_id not in {o.id for o in req.options}:
        return f"Unknown option. Valid: {', '.join(o.id for o in req.options)}"
    label = next(o.label for o in req.options if o.id == option_id)
    post("group", {"t": "note", "text": f"👤 Decision relayed by the Chief of Staff: **{label}**"},
         log_line=f"DUTY OFFICER DECISION: {label}")
    S.oc_pending = None
    return pump(option_id)


def t_ask_specialist(agent: str, question: str) -> str:
    post("group", {"t": "msg", "who": "MODERATOR", "text": f"@{agent.title()} {question}", "via": True},
         log_line=f"MODERATOR to {agent}: {question}")
    return talk(agent, "group", question, asked_by="the Chief of Staff (in #ops-team)")


def t_get_record() -> str:
    lines = [ln for ln in S.oc_group_log if ln and not ln.startswith(("MODERATOR:", "ATC:", "PILOT:", "GROUND:"))]
    return "\n".join(lines[-80:]) or "No negotiation yet."


def t_compare_plans(plans: list) -> str:
    s = scenario()
    if s is None:
        return "No flight loaded."
    out = []
    for raw in plans[:8]:
        p = Proposal(**raw)
        if s.runway(p.runway) is None or s.gate(p.gate) is None:
            out.append({"plan": raw, "valid": False, "reason": "unknown runway or stand"})
            continue
        r = check_all(s, p)
        row = {"plan": raw, "tldt": s.clock(p.landing_slot_min), "safe": is_safe(r)}
        if row["safe"]:
            row["usd_total"] = cost(s, p, r)
            row["usd_items"] = {k: round(v) for k, v in cost_breakdown(s, p, r).items() if v}
        else:
            row["hard_violations"] = [f"{d}: {c.name}" for d, x in r.items() for c in x.violations()]
        out.append(row)
    return json.dumps(out)


def fire_emergency(kind: str, reason: str = "", by: str = "duty officer") -> str:
    """MAYDAY fast path: deterministic plan first (milliseconds), broadcast, then optional readbacks."""
    s = scenario()
    if s is None:
        return "No flight loaded."
    plan = plan_emergency(s, kind, reason)
    if S.oc_session is not None and S.oc_outcome is None:
        post("group", {"t": "note", "text": "⛔ Negotiation aborted: emergency declared."},
             log_line="SYSTEM: negotiation aborted by emergency")
    S.oc_session, S.oc_pending, S.oc_ratify, S.oc_emergency = None, None, None, plan
    alarm = dispatch(Alarm(code="MAYDAY", severity="ALARM", title=f"{KIND_LABEL[kind]} {s.flight}",
                           detail=(f"Divert {plan.divert_to}" if plan.divert_to else
                                   f"RWY {plan.proposal.runway} {plan.landing_clock}, stand {plan.proposal.gate}")
                           + (f"; {reason}" if reason else ""), flight=s.flight, turn=0))
    log = ConsensusLog()
    log.add(0, by.upper().replace(" ", "_"), "emergency_declared", {"kind": kind, "reason": reason})
    log.add(0, "SYSTEM", "emergency_plan", plan.model_dump())
    log.add(0, "SYSTEM", "alarm", alarm.model_dump())
    log.save(ROOT / "logs" / f"EMERGENCY_{s.flight}_{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    post("group", {"t": "emergency", "data": plan, "live": True},
         log_line=f"EMERGENCY {kind} declared by {by}: " + (f"divert {plan.divert_to}" if plan.divert_to else
                  f"RWY {plan.proposal.runway} {plan.landing_clock} stand {plan.proposal.gate}"))
    for a in ("ATC", "PILOT", "GROUND"):
        show_status(a, "working")
        post(a, {"t": "msg", "who": a, "text": f"🚨 **{KIND_LABEL[kind]} {s.flight}**. My actions:\n- "
                 + "\n- ".join(plan.actions[a])})
    if engine == ENGINES[0]:  # readbacks never delay the plan: it is already on everyone's screen
        with st.spinner("Desks reading back…"):
            rb = readbacks(plan, make_engine_agents(engine, model, effort))
        for a, text in rb.items():
            if text:
                post("group", {"t": "msg", "who": a, "text": f"📻 {text}", "meta": "readback"},
                     log_line=f"{a} readback: {text}")
    for a in ("ATC", "PILOT", "GROUND"):
        show_status(a, "idle")
    return (f"Emergency plan issued in {plan.decision_ms:.2f} ms: "
            + (f"divert to {plan.divert_to}" if plan.divert_to else
               f"RWY {plan.proposal.runway} at {plan.landing_clock}, stand {plan.proposal.gate}, {plan.fuel_status}")
            + f"; displaced traffic: {', '.join(plan.displaced) or 'none'}; alarm via {', '.join(alarm.channels)}.")


def t_declare_emergency(kind: str, reason: str) -> str:
    return fire_emergency(kind, reason, by="Chief of Staff")


CHIEF_IMPL = {"compare_plans": t_compare_plans, "declare_emergency": t_declare_emergency,
              "find_flights": t_find_flights, "load_flight": t_load_flight, "flight_status": t_flight_status,
              "start_negotiation": t_start_negotiation, "resolve_decision": t_resolve_decision,
              "ask_specialist": t_ask_specialist, "get_negotiation_record": t_get_record}

# ============================================================================
# Stop / steer / override a live negotiation
# ============================================================================
def negotiation_active() -> bool:
    return S.oc_session is not None and S.oc_outcome is None


def restart(new_s, agents, autonomy: str, directive: str, label: str) -> None:
    sess = NegotiationSession(new_s, agents, Autonomy(autonomy))
    if directive:  # every agent sees it on every turn: the constraint ledger is always on their bus
        sess.ledger[("CHIEF", "operator_directive")] = f"CHIEF OF STAFF DIRECTIVE (from the duty officer): {directive}"
        sess.log.add(0, "HUMAN", "operator_directive", {"directive": directive})
    S.oc_session, S.oc_outcome, S.oc_pending, S.oc_ratify, S.oc_halted = sess, None, None, None, False
    post("group", {"t": "note", "text": f"↻ {label}"}, log_line=f"SYSTEM: {label}")
    pump()


def steer_negotiation(text: str, override: bool = False) -> None:
    """Chief synthesizes -> firewall (code) -> apply events to the flight -> restart from turn 1."""
    s = scenario()
    S.oc_running, S.oc_halted = False, True  # freeze the current negotiation first
    for a in ("ATC", "PILOT", "GROUND"):
        show_status(a, "idle")
    show_status("MODERATOR", "working")
    chief = make_agents(model=model, effort=effort)["MODERATOR"]
    with st.spinner("Chief of Staff is synthesizing the new information…"):
        d = steer.synthesize(chief, s, text, t_get_record())
    if d.emergency != "NONE":
        show_status("MODERATOR", "idle")
        fire_emergency(d.emergency, d.summary, by="duty officer (steer)")
        return
    accepted, refused, notes = steer.firewall(s, d)
    if refused:
        with st.spinner("Safety firewall refused part of it: the Chief is explaining…"):
            r = steer.explain_refusal(chief, text, refused)
        show_status("MODERATOR", "waiting")
        post("group", {"t": "note", "text": "🛡️ **Safety firewall refused** this: it would relax "
                       + ", ".join(steer.LIMIT_LABEL[x] for x in refused) + ". The negotiation stays stopped."},
             log_line=f"FIREWALL refused relaxing {refused}")
        post("group", {"t": "msg", "who": "MODERATOR", "text": f"**{r.header}.** {r.explanation}\n\n"
                       f"Safe alternative you can send: _{r.safe_alternative}_"},
             log_line=f"MODERATOR: {r.header}: {r.explanation} Alternative: {r.safe_alternative}")
        return
    new_s = steer.apply(s, accepted, d.summary)
    S.oc_scenario = new_s
    applied = [steer.describe(new_s, e) for e in accepted]
    directive = d.summary + (" Applied: " + "; ".join(applied) + "." if applied else "") + "".join(
        f" {g.agent}: {g.instruction}" for g in d.guidance)
    show_status("MODERATOR", "idle")
    post("group", {"t": "msg", "who": "MODERATOR", "text": f"🧭 **New information synthesized.** {d.summary}"
                   + ("\n\n**Applied to the flight (the safety checks now enforce it):**\n- " + "\n- ".join(applied)
                      if applied else "")
                   + ("\n\n**Guidance:**\n- " + "\n- ".join(f"{g.agent}: {g.instruction}" for g in d.guidance)
                      if d.guidance else "")
                   + (f"\n\n_Ignored: {'; '.join(notes)}_" if notes else "")},
         log_line=f"MODERATOR directive: {directive}")
    if override:
        restart(new_s, make_rule_agents(), "AUTONOMOUS", directive,
                "Override: the deterministic layer decides on the updated flight (no AI)")
    else:
        restart(new_s, make_engine_agents(engine, model, effort), default_autonomy, directive,
                "Negotiation restarted from turn 1 with the new information")


# ============================================================================
# Page
# ============================================================================
s = scenario()
hero(compact=True)
if s is not None:
    fids_board(s, S.oc_outcome, emergency=S.oc_emergency, running=S.oc_session is not None and S.oc_outcome is None,
               pending=S.oc_pending is not None or S.oc_ratify is not None)
if channel == "group":
    st.subheader("👥 #ops-team")
    st.caption("You, the Chief of Staff and the three specialists. Messages go to the Chief unless you @mention "
               "@ATC, @Pilot, @Ground or @Chief. Negotiations run here.")
else:
    st.subheader(f"✉️ {NAMES[channel]}")
    st.caption(f"Private DM with {NAMES[channel]}. Only you and this agent see it; it keeps its own memory "
               "and is aware of what happens in #ops-team.")

if s is not None:
    with st.expander(f"✈️ Active flight: {s.flight} {s.origin}→{s.destination} {s.date} · SIBT {s.clock(0)} · "
                     f"WX {s.weather_delay_min} · LATE {s.late_aircraft_delay_min}"):
        weather_card(s)
        out = S.oc_outcome
        st.plotly_chart(timeline(s, out.proposal if out else None, out.reference.proposal if out else None),
                        width="stretch", key=f"tl_{channel}")
elif not S.oc_items["group"]:
    st.info("Try: *“Find a weather-delayed flight into ORD and load it”*, then *“Run the negotiation, supervised”*. "
            "Or DM a specialist: *“How do crosswind limits affect runway choice?”*", icon="💡")

for item in S.oc_items[channel]:
    draw(item)

# Stakeholder votes on an agreed plan (actionable in #ops-team; flagged everywhere)
if S.oc_ratify is not None:
    if channel == "group":
        votes = ratification_form(S.oc_ratify, "oc_ratify")
        if votes is not None:
            S.oc_votes, S.oc_ratify = votes, None
            st.rerun()
    else:
        st.warning(f"🔴 Stakeholder votes are waiting in #ops-team: plan #{S.oc_ratify.plan_id}")

if S.oc_votes is not None:
    votes, S.oc_votes = S.oc_votes, None
    pump(votes)
    st.rerun()

# Pending decision card (actionable in #ops-team; flagged everywhere)
req = S.oc_pending
if req is not None:
    if channel == "group":
        colour = {"INFO": "blue", "WARNING": "orange", "ALARM": "red"}[req.severity]
        with st.container(border=True):
            st.markdown(f"#### :{colour}[👤 Your decision · {req.title}]")
            for line in req.briefing:
                st.markdown(f"- {line}")
            for opt in req.options:
                c1, c2 = st.columns([1, 3])
                if c1.button(opt.label, key=f"oc_opt_{req.code}_{req.turn}_{opt.id}",
                             type="primary" if opt.id == req.default else "secondary"):
                    S.oc_click = opt.id
                    st.rerun()
                c2.caption(opt.description)
            st.caption("…or just tell the Chief of Staff what you want in the chat.")
    else:
        st.warning(f"🔴 A decision is waiting for you in #ops-team: {req.title}")

if S.oc_click is not None and req is not None:
    choice, S.oc_click = S.oc_click, None
    label = next(o.label for o in req.options if o.id == choice)
    post("group", {"t": "msg", "who": "user", "text": f"Decision: **{label}**"}, log_line=f"DUTY OFFICER DECISION: {label}")
    S.oc_pending = None
    pump(choice)
    st.rerun()

if S.get("mayday_request"):
    req_m = S.pop("mayday_request")
    if scenario() is None:
        st.toast("MAYDAY needs an active flight: load one first (ask the Chief of Staff).", icon="⚠️")
    else:
        S.oc_channel = "group"
        post("group", {"t": "msg", "who": "user", "text": f"🚨 **MAYDAY CONFIRMED**: {KIND_LABEL[req_m['kind']]}"
                       + (f" ({req_m['reason']})" if req_m["reason"] else "")},
             log_line=f"DUTY OFFICER: MAYDAY {req_m['kind']} {req_m['reason']}")
        fire_emergency(req_m["kind"], req_m["reason"])
        st.rerun()

demo_prompt = S.pop("demo_prompt", None)  # sent by the Demo button in Ops settings

# Voice: talk to the agents (recorded in the browser, transcribed locally with Whisper)
voice_text = None
if not S.get("voice_enabled"):
    if not S.get("voice_dismissed"):
        with st.container(border=True):
            st.markdown("🎙️ **Talk to the agents with your voice?**")
            st.caption("When you press the microphone, your browser will ask to **allow microphone access**: choose "
                       "Allow. On a Mac, if it never asks, open System Settings → Privacy & Security → Microphone "
                       "and switch your browser on. Speech recognition runs **locally on this computer** (Whisper, "
                       "offline): no audio leaves your machine, so no macOS Speech Recognition permission is needed. "
                       "The first time downloads the ~140 MB speech model.")
            v1, v2, _ = st.columns([1.2, 1, 4])
            if v1.button("🎙️ Enable voice", type="primary"):
                with st.spinner("Loading the local speech model (first time downloads ~140 MB)…"):
                    voice.warm_up()
                S.voice_enabled = True
                st.rerun()
            if v2.button("Not now"):
                S.voice_dismissed = True
                st.rerun()
else:
    rec = st.audio_input("🎙️ Speak to " + ("the team (say “ATC…”, “Captain…”, or just talk to the Chief)"
                                          if channel == "group" else NAMES[channel]), key=f"mic_{channel}")
    if rec is not None:
        fp = voice.fingerprint(rec.getvalue())
        if fp != S.get("voice_last"):
            S.voice_last = fp
            with st.spinner("Transcribing locally…"):
                voice_text = voice.transcribe(rec.getvalue())
            if not voice_text:
                st.toast("Didn't catch that: try again a little closer to the mic.", icon="🎙️")

steer_mode = "Ask"
if channel == "group":  # always on screen in the group chat
    with st.container(border=True):
        LABELS = {"Steer": "🧭 Steer: re-plan with this", "Ask": "💬 Ask: just chat",
                  "Override": "⛔ Override: deterministic layer decides"}
        if S.oc_scenario is None:
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown("⚪ **No flight loaded.** Ask the Chief of Staff to load one; then you can stop or steer "
                        "its negotiation from here at any time.")
            c2.button("⏹ Stop", width="stretch", disabled=True, key="oc_stop_idle")
        elif S.get("oc_halted"):
            st.markdown("⏸️ **Negotiation stopped.** Type what changed below and send it as **Steer** (the Chief "
                        "synthesizes it and restarts the negotiation), or:")
            h1, h2, _ = st.columns([1, 1.6, 3])
            h1.button("▶ Resume", on_click=lambda: S.update(oc_halted=False, oc_running=True))
            h2.button("⛔ Deterministic layer decides", on_click=lambda: S.update(oc_override_now=True))
            steer_mode = st.segmented_control("When I send a message", list(LABELS), default="Steer",
                                              key="oc_send_mode_live", format_func=LABELS.get) or "Steer"
        elif negotiation_active():
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown("🟢 **Live negotiation.** Something changed? Stop it, or type it below to steer.")
            c2.button("⏹ Stop", type="primary", width="stretch",
                      on_click=lambda: S.update(oc_halted=True, oc_running=False))
            steer_mode = st.segmented_control("When I send a message", list(LABELS), default="Steer",
                                              key="oc_send_mode_live", format_func=LABELS.get) or "Steer"
        else:
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown(f"⚪ **No negotiation running** for {S.oc_scenario.flight}. Choose **Steer** to send new "
                        "information that updates the flight and starts a negotiation with it.")
            c2.button("⏹ Stop", width="stretch", disabled=True, key="oc_stop_idle")
            steer_mode = st.segmented_control("When I send a message", list(LABELS), default="Ask",
                                              key="oc_send_mode_idle", format_func=LABELS.get) or "Ask"

if S.pop("oc_override_now", False):
    S.oc_halted = False
    restart(scenario(), make_rule_agents(), "AUTONOMOUS", "",
            "Override: the deterministic layer decides (no AI)")
    st.rerun()

placeholder = ("Message the team… (@ATC, @Pilot, @Ground to address someone)" if channel == "group"
               else f"Message {NAMES[channel]} privately…")
prompt = st.chat_input(placeholder, disabled=len(os.environ.get("ANTHROPIC_API_KEY", "")) < 40)
if demo_prompt:
    prompt = demo_prompt
if voice_text and not prompt:
    prompt = voice_text
    # spoken names work like @mentions in the group chat
    for spoken, tag in (("atc", "@ATC"), ("captain", "@Captain"), ("pilot", "@Pilot"), ("ground", "@Ground"),
                        ("aocc", "@AOCC"), ("chief", "@Chief")):
        if channel == "group" and re.match(rf"^\W*{spoken}\b", prompt, re.I):
            prompt = f"{tag} {prompt}"
            break
if prompt:
    post(channel, {"t": "msg", "who": "user", "text": prompt, "meta": "🎙️ spoken" if voice_text else None},
         log_line=f"DUTY OFFICER: {prompt}")
    if channel == "group" and steer_mode in ("Steer", "Override") and S.oc_scenario is not None:
        steer_negotiation(prompt, override=steer_mode == "Override")
        st.rerun()
    if channel == "group":
        targets = list(dict.fromkeys(MENTIONS[m.lower()] for m in re.findall(r"@\w+", prompt)
                                     if m.lower() in MENTIONS)) or ["MODERATOR"]
        for agent in targets:
            talk(agent, "group", prompt)
    else:
        talk(channel, channel, prompt)
    st.rerun()

# A negotiation interrupted by any click (switching channel, a sidebar widget...) resumes where it stopped.
if (S.oc_session is not None and S.get("oc_running") and not S.get("oc_halted") and S.oc_outcome is None
        and S.oc_pending is None and S.oc_ratify is None):
    pump()
    st.rerun()
