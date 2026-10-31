"""Realtime API for a React (or any) dashboard.   uvicorn api:app --reload --port 8000

REST for actions, one WebSocket (/ws) that pushes EVERYTHING as it happens:
  weather.alert / weather.board     automated safety checks (rain, lightning, wind, visibility, wind shear, ...)
  negotiation.* / agent.* / chief.* the 3-turn negotiation, each agent's message with its thinking summary
  decision.required                 the run is paused and needs the duty officer
  alarm / emergency.declared        escalations
Every event: {"id": n, "type": "...", "ts": "ISO-8601 UTC", "data": {...}}. GET /api/events?since=n replays.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config  # noqa: F401  (loads .env)
import ops_settings
import profiles
import watch
from agents import make_agents
from baseline_agents import make_drill_agents, make_rule_agents
from constraints import cost_breakdown, plan_robustness
from emergency import KIND_LABEL, plan_emergency
from escalation import Alarm, dispatch, push
from negotiation import Autonomy, NegotiationSession
from scenario import Scenario, build_scenario, load_disrupted_flights


# ============================================================================
# Event hub
# ============================================================================

class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.history: list[dict] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.lock = threading.Lock()
        self.n = 0

    def emit(self, type_: str, data: Any) -> None:
        """Thread-safe: callable from worker threads."""
        with self.lock:
            self.n += 1
            ev = {"id": self.n, "type": type_, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "data": data}
            self.history = (self.history + [ev])[-500:]
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._send(ev), self.loop)

    async def _send(self, ev: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(ev)
            except Exception:  # noqa: BLE001 - a dropped client must not affect others
                self.clients.discard(ws)


hub = Hub()


# ============================================================================
# State
# ============================================================================

class Run(BaseModel, arbitrary_types_allowed=True):
    id: str
    status: Literal["running", "paused", "done"] = "running"
    pending: Optional[dict] = None


STATE: dict[str, Any] = {"scenario": None, "runs": {}, "choices": {}, "emergency": None}


def flight_summary(s: Scenario) -> dict:
    return {"flight": s.flight, "airline": s.airline, "tail": s.tail_number, "origin": s.origin,
            "destination": s.destination, "destination_name": s.destination_name, "date": s.date,
            "sibt": s.clock(0), "eldt": s.clock(s.eta_min), "latest_safe_landing": s.clock(s.latest_safe_landing),
            "weather_delay_min": s.weather_delay_min, "late_aircraft_delay_min": s.late_aircraft_delay_min,
            "aircraft": s.aircraft_type, "weather_mode": s.weather_mode, "weather": s.weather_attribution,
            "runways": [{"id": r.id, "crosswind_kt": r.crosswind_kt, "status": r.closure_reason} for r in s.runways]}


def plan_json(s: Scenario, p) -> Optional[dict]:
    if p is None:
        return None
    return {"landing_slot_min": p.landing_slot_min, "runway": p.runway, "gate": p.gate,
            "tldt": s.clock(p.landing_slot_min) if p.runway != "DIVERT" else None}


def event_json(s: Scenario, kind: str, d: Any) -> tuple[str, dict]:
    """Translate engine events to the public event contract."""
    if kind == "turn":
        return "negotiation.turn", {"turn": d["turn"]}
    if kind == "thinking":
        return "agent.working", {"agents": d["agents"], "turn": d["turn"]}
    if kind == "message":
        m = d.metric
        return "agent.message", {
            "agent": d.agent, "turn": d.turn, "stance": d.stance, "plan_id": d.plan_id, "answering_plan": d.answering_plan,
            "proposal": plan_json(s, d.proposal), "rationale": d.rationale, "concession": d.concession,
            "constraints": [c.model_dump() for c in d.constraints], "thinking": d.thinking,
            "metric": m.model_dump(include={"latency_s", "input_tokens", "output_tokens", "cost_usd"}) if m else None}
    if kind == "brief":
        return "chief.brief", {"turn": d.turn, "summary": d.summary, "open_conflicts": d.open_conflicts,
                               "bridge_proposal": plan_json(s, d.bridge_proposal),
                               "directives": [x.model_dump() for x in d.directives], "thinking": d.thinking}
    if kind == "system":
        return "system.note", d
    if kind == "validation":
        return "validation", {"turn": d["turn"], "plan_id": d["plan_id"], "domains": {
            dom: {"hard_ok": r.hard_ok, "checks": [c.model_dump(exclude={"domain"}) for c in r.checks]}
            for dom, r in d["reports"].items()}}
    if kind == "alarm":
        return "alarm", d.model_dump()
    if kind == "decision":
        return "decision.required", {"code": d.code, "severity": d.severity, "title": d.title, "turn": d.turn,
                                     "briefing": d.briefing, "default": d.default,
                                     "options": [{"id": o.id, "label": o.label, "description": o.description}
                                                 for o in d.options]}
    if kind == "decision_auto":
        return "decision.auto", {"code": d["request"].code, "choice": d["choice"]}
    if kind == "metric":
        return "agent.metric", d.model_dump()
    if kind == "outcome":
        o = d
        breakdown = ({k: round(v) for k, v in cost_breakdown(s, o.proposal, o.reports).items() if v}
                     if o.reports and all(r.hard_ok for r in o.reports.values()) else {})
        others = [o.reference.proposal] + [a for a, _ in o.reference.alternatives]
        return "negotiation.outcome", {
            "status": o.status, "plan_source": o.plan_source, "turns": o.turns_used, "plan": plan_json(s, o.proposal),
            "cost_usd": o.cost, "optimum_cost_usd": o.reference.cost, "cost_breakdown": breakdown,
            "robustness": plan_robustness(s, o.proposal, [x for x in others if x != o.proposal]) if breakdown else None,
            "usage": o.usage, "alarms": len(o.alarms), "log_verified": o.log_verified, "elapsed_s": o.elapsed_s}
    return kind, {}


def run_negotiation(run_id: str, session: NegotiationSession, s: Scenario) -> None:
    run: Run = STATE["runs"][run_id]
    choice = None
    while True:
        kind, d = session.advance(choice)
        choice = None
        typ, payload = event_json(s, kind, d)
        hub.emit(typ, {"run_id": run_id, **payload})
        if kind == "decision":
            run.status, run.pending = "paused", payload
            choice = STATE["choices"][run_id].get()  # blocks this worker only, until POST .../decision
            run.status, run.pending = "running", None
        if kind == "outcome":
            run.status = "done"
            return


# ============================================================================
# App
# ============================================================================

async def weather_loop() -> None:
    while True:
        try:
            events = await asyncio.to_thread(watch.poll)
            for ev in events:
                hub.emit("weather.alert", ev.model_dump())
            hub.emit("weather.board", {"alerts": [a.model_dump() for a in watch.active_alerts()]})
        except Exception as e:  # noqa: BLE001 - the watch must keep running
            hub.emit("weather.error", {"error": f"{type(e).__name__}: {e}"})
        await asyncio.sleep(max(60, ops_settings.get().watch_interval_s))


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    task = asyncio.create_task(weather_loop())
    yield
    task.cancel()


app = FastAPI(title="A-CDM Multi-Agent Ops API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    hub.clients.add(socket)
    try:
        for ev in hub.history[-50:]:  # catch up a late joiner
            await socket.send_json(ev)
        while True:
            await socket.receive_text()  # keep-alive; clients may send "ping"
    except WebSocketDisconnect:
        hub.clients.discard(socket)


@app.get("/api/events")
def events(since: int = 0):
    return [e for e in hub.history if e["id"] > since]


@app.get("/api/state")
def state():
    s = STATE["scenario"]
    runs = STATE["runs"]
    return {"flight": flight_summary(s) if s else None,
            "runs": [r.model_dump() for r in runs.values()],
            "alerts": [a.model_dump() for a in watch.active_alerts()],
            "settings": ops_settings.get().model_dump(),
            "agents": {k: {"name": v["name"], "avatar": v["avatar"]} for k, v in profiles.load().items()
                       if k in ("ATC", "PILOT", "GROUND", "MODERATOR")}}


@app.get("/api/flights")
def flights(airport: str = "", bucket: str = "any", min_weather_delay: int = 0, limit: int = 20):
    df = load_disrupted_flights(limit=None)
    if airport:
        df = df[df.DESTINATION_AIRPORT == airport.upper()]
    if bucket != "any":
        df = df[df.BUCKET == bucket]
    if min_weather_delay:
        df = df[df.WEATHER_DELAY >= min_weather_delay]
    return [{"row_id": int(i), "flight": f"{r.AIRLINE}{r.FLIGHT_NUMBER}", "origin": r.ORIGIN_AIRPORT,
             "destination": r.DESTINATION_AIRPORT, "date": f"2015-{r.MONTH:02d}-{r.DAY:02d}",
             "scheduled_arrival": f"{int(r.SCHEDULED_ARRIVAL):04d}", "weather_delay_min": int(r.WEATHER_DELAY),
             "late_aircraft_delay_min": int(r.LATE_AIRCRAFT_DELAY), "bucket": r.BUCKET}
            for i, r in df.head(min(limit, 100)).iterrows()]


@app.post("/api/flights/{row_id}/load")
def load_flight(row_id: int, weather: Literal["HISTORICAL", "LIVE", "OFF"] = "HISTORICAL"):
    df = load_disrupted_flights(limit=None)
    if row_id not in df.index:
        raise HTTPException(404, "unknown row_id")
    s = build_scenario(df.loc[row_id], weather_mode=weather)
    STATE.update(scenario=s, emergency=None)
    hub.emit("flight.loaded", flight_summary(s))
    return flight_summary(s)


class StartNegotiation(BaseModel):
    autonomy: Literal["AUTONOMOUS", "SUPERVISED", "STEP_BY_STEP"] = "SUPERVISED"
    engine: Literal["claude", "rules", "drill"] = "claude"
    effort: Literal["low", "medium", "high"] = "medium"


@app.post("/api/negotiations")
def start(body: StartNegotiation):
    s = STATE["scenario"]
    if s is None:
        raise HTTPException(409, "load a flight first")
    agents = {"claude": lambda: make_agents(effort=body.effort), "rules": make_rule_agents,
              "drill": make_drill_agents}[body.engine]()
    run_id = uuid.uuid4().hex[:8]
    session = NegotiationSession(s, agents, Autonomy(body.autonomy), deadline_s=ops_settings.get().time_budget_s)
    STATE["runs"][run_id] = Run(id=run_id)
    STATE["choices"][run_id] = queue.Queue()
    hub.emit("negotiation.started", {"run_id": run_id, "flight": s.flight, **body.model_dump()})
    threading.Thread(target=run_negotiation, args=(run_id, session, s), daemon=True).start()
    return {"run_id": run_id}


class Decision(BaseModel):
    option_id: str


@app.post("/api/negotiations/{run_id}/decision")
def decide(run_id: str, body: Decision):
    run = STATE["runs"].get(run_id)
    if run is None or run.status != "paused":
        raise HTTPException(409, "no decision pending for this run")
    if body.option_id not in {o["id"] for o in run.pending["options"]}:
        raise HTTPException(422, "unknown option")
    hub.emit("decision.made", {"run_id": run_id, "option_id": body.option_id, "by": "HUMAN"})
    STATE["choices"][run_id].put(body.option_id)
    return {"ok": True}


class Emergency(BaseModel):
    kind: Literal["FUEL", "MEDICAL", "TECHNICAL"]
    reason: str = ""


@app.post("/api/emergency")
def emergency(body: Emergency):
    s = STATE["scenario"]
    if s is None:
        raise HTTPException(409, "load a flight first")
    plan = plan_emergency(s, body.kind, body.reason)
    alarm = dispatch(Alarm(code="MAYDAY", severity="ALARM", title=f"{KIND_LABEL[body.kind]} {s.flight}",
                           detail=f"Divert {plan.divert_to}" if plan.divert_to else
                           f"RWY {plan.proposal.runway} {plan.landing_clock}, stand {plan.proposal.gate}",
                           flight=s.flight, turn=0))
    payload = {**plan.model_dump(exclude={"proposal"}), "label": KIND_LABEL[body.kind], "plan": plan_json(s, plan.proposal),
               "channels": alarm.channels}
    STATE["emergency"] = payload
    hub.emit("emergency.declared", payload)
    return payload


@app.get("/api/settings")
def get_settings():
    return ops_settings.get().model_dump()


@app.put("/api/settings")
def put_settings(body: dict):
    merged = ops_settings.OpsSettings(**{**ops_settings.get().model_dump(), **body})
    ops_settings.save(merged)
    hub.emit("settings.updated", merged.model_dump())
    return merged.model_dump()


@app.get("/api/alerts")
def alerts():
    return {"active": [a.model_dump() for a in watch.active_alerts()],
            "recent": [e.model_dump() for e in watch.recent_events(50)]}


@app.post("/api/alerts/check")
async def check_now():
    events = await asyncio.to_thread(watch.poll)
    for ev in events:
        hub.emit("weather.alert", ev.model_dump())
    hub.emit("weather.board", {"alerts": [a.model_dump() for a in watch.active_alerts()]})
    return {"changes": len(events)}


@app.post("/api/alerts/test")
def test_alert():
    alert = watch.Alert(airport="TEST", check="thunderstorm", severity="ALARM", title="Test: thunderstorm at the field",
                        detail="Test alert: if you see and hear this, notifications work.",
                        observed=datetime.now(timezone.utc).strftime("%H:%MZ"))
    channels = push("🧪 TEST ALARM: thunderstorm at the field", alert.detail, "ALARM")
    hub.emit("weather.alert", {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": "NEW",
                               "alert": alert.model_dump(), "channels": channels, "test": True})
    return {"channels": channels}
