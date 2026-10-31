"""Turn-bounded negotiation engine with human-in-the-loop gates, escalation and a tamper-evident log.

Protocol (max 3 turns). The Chief of Staff opens; then, per turn:
  1. ATC tables plan #n (PROPOSE). ATC's deterministic validator gates it; re-tabling an already-countered
     plan unchanged is a DEADLOCK.
  2. Flight Crew and AOCC answer in parallel (independent Claude calls) with ACCEPT or COUNTER, citing #n.
     Replies citing an older plan number are discarded (stale-state guard).
  3. Safety gate: an ACCEPT that breaks the agent's own hard limits is overridden.
  4. Consensus = both ACCEPT plan #n AND it passes every hard check in every domain.
  5. No consensus: the Chief of Staff briefs the next turn.
After turn 3 the deterministic non-AI planner decides.

The engine is a generator, so it can PAUSE for a human at decision points without losing any state:
  AUTONOMOUS    never pauses; every decision takes its safe default (alarms still go out)
  SUPERVISED    pauses on risk: agreed plan near a limit, deadlock, agent failure/timeout, fallback
  STEP_BY_STEP  pauses after every turn and before committing anything
Whatever a human picks, only plans that pass every hard check are ever offered.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Literal, Optional

from pydantic import BaseModel

from agents import AgentError, AgentMessage, BaseAgent, ModeratorAgent, ModeratorMessage, render_brief, render_message
from constraints import (
    VALIDATORS, Domain, DomainReport, FallbackResult, Proposal, check_all, cost, cost_breakdown, deterministic_plan,
    is_safe,
)
from escalation import Alarm, dispatch
from monitoring import CallMetric, summarise
import ops_settings
from scenario import Scenario

MAX_TURNS = 3
LOG_DIR = Path(__file__).parent / "logs"


class Autonomy(str, Enum):
    AUTONOMOUS = "AUTONOMOUS"
    SUPERVISED = "SUPERVISED"
    STEP_BY_STEP = "STEP_BY_STEP"


# ============================================================================
# Hash-chained consensus log
# ============================================================================

class LogEntry(BaseModel):
    seq: int
    ts: str
    turn: int
    actor: str
    kind: str
    payload: dict
    prev_hash: str
    hash: str


class ConsensusLog:
    """Append-only log; each entry carries the SHA-256 of the previous one, so any edit breaks the chain."""

    def __init__(self) -> None:
        self.entries: list[LogEntry] = []

    def add(self, turn: int, actor: str, kind: str, payload: dict) -> LogEntry:
        prev = self.entries[-1].hash if self.entries else "GENESIS"
        body = {"seq": len(self.entries), "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "turn": turn, "actor": actor, "kind": kind, "payload": payload, "prev_hash": prev}
        entry = LogEntry(**body, hash=hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest())
        self.entries.append(entry)
        return entry

    def verify(self) -> bool:
        prev = "GENESIS"
        for e in self.entries:
            body = e.model_dump(exclude={"hash"})
            if e.prev_hash != prev or hashlib.sha256(
                    json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != e.hash:
                return False
            prev = e.hash
        return True

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(e.model_dump_json() for e in self.entries) + "\n")
        return path


# ============================================================================
# Human decision points
# ============================================================================

class DecisionOption(BaseModel):
    id: str
    label: str
    description: str
    proposal: Optional[Proposal] = None


class DecisionRequest(BaseModel):
    code: str  # CONSENSUS_REVIEW, TURN_REVIEW, DEADLOCK, AGENT_TIMEOUT, AGENT_FAILURE, FALLBACK_REVIEW
    severity: Literal["INFO", "WARNING", "ALARM"]
    title: str
    briefing: list[str]  # the "clear box": what happened and why this needs a human
    turn: int
    options: list[DecisionOption]
    default: str  # the option AUTONOMOUS mode takes


class DecisionRecord(BaseModel):
    code: str
    turn: int
    option: str
    by: Literal["HUMAN", "AUTO"]


class _Stop(Exception):
    def __init__(self, choice: str, plan: Optional[Proposal] = None):
        self.choice, self.plan = choice, plan


# ============================================================================
# Outcome
# ============================================================================

class Outcome(BaseModel):
    status: Literal["CONSENSUS", "FALLBACK", "HUMAN_DECISION"]
    plan_source: str
    turns_used: int
    proposal: Proposal
    reports: dict[str, DomainReport]
    cost: Optional[float]
    reference: FallbackResult
    reason: Optional[str] = None
    messages: list[AgentMessage]
    briefs: list[ModeratorMessage]
    decisions: list[DecisionRecord]
    alarms: list[Alarm]
    metrics: list[CallMetric]
    usage: dict
    deadlock: bool
    deadline_hit: bool = False
    safety_overrides: int
    stale_replies: int
    log_path: Optional[str] = None
    log_verified: bool
    elapsed_s: float


Event = tuple[str, object]


def _key(p: Proposal) -> tuple:
    return (p.landing_slot_min, p.runway, p.gate)


class NegotiationSession:
    """Drive with `advance()`: returns the next (kind, data) event. When kind == "decision", call
    `advance(option_id)` with the human's choice. kind == "outcome" ends the session."""

    def __init__(self, s: Scenario, agents: dict[str, BaseAgent], autonomy: Autonomy = Autonomy.AUTONOMOUS,
                 save_log: bool = True, deadline_s: Optional[float] = None):
        self.s, self.agents, self.autonomy, self.save_log = s, agents, Autonomy(autonomy), save_log
        self.deadline_s = deadline_s  # hard wall-clock budget for the whole negotiation (time-critical ops)
        self.deadline_hit = False
        self.log = ConsensusLog()
        self.messages: list[AgentMessage] = []
        self.briefs: list[ModeratorMessage] = []
        self.decisions: list[DecisionRecord] = []
        self.alarms: list[Alarm] = []
        self.metrics: list[CallMetric] = []
        self.public: list[tuple[int, str]] = []  # (turn, rendered line) for the compact bus view
        self.ledger: dict[tuple[str, str], str] = {}  # (agent, constraint) -> rendered, latest wins
        self.seen_plans: dict[tuple, Proposal] = {}
        self.deadlock = False
        self.safety_overrides = 0
        self.stale_replies = 0
        self.started = time.monotonic()
        self.reference = deterministic_plan(s)
        self.outcome: Optional[Outcome] = None
        self._gen = self._run()
        self._primed = False

    # ------------------------------------------------------------------ driving
    def advance(self, choice: Optional[str] = None) -> Event:
        if not self._primed:
            self._primed = True
            return next(self._gen)
        return self._gen.send(choice)

    # ------------------------------------------------------------------ helpers
    def _bus(self, turn: int) -> str:
        ledger = "\n".join(f"  {v}" for v in self.ledger.values()) or "  (none yet)"
        recent = "\n".join(line for t, line in self.public if t >= turn - 1)
        return f"CONSTRAINT LEDGER (latest per specialist):\n{ledger}\nRECENT TRAFFIC:\n{recent or '  (none)'}"

    def _record(self, msg: AgentMessage):
        self.messages.append(msg)
        for c in msg.constraints:
            self.ledger[(msg.agent, c.name)] = f"{msg.agent} {c.name}={c.value}{' [HARD]' if c.hard else ''}"
        self.public.append((msg.turn, render_message(msg, self.s)))
        self.seen_plans.setdefault(_key(msg.proposal), msg.proposal)
        self.log.add(msg.turn, msg.agent, "message", msg.model_dump(exclude={"metric"}))
        return ("message", msg)

    def _note(self, turn: int, kind: str, text: str, payload: Optional[dict] = None):
        self.public.append((turn, f"[T{turn}] SYSTEM {kind}: {text}"))
        self.log.add(turn, "SYSTEM", kind, {"text": text, **(payload or {})})
        return ("system", {"turn": turn, "kind": kind, "text": text})

    def _metric(self, m: Optional[CallMetric]):
        if m is not None:
            self.metrics.append(m)
            self.log.add(m.turn, m.agent, "call_metric", m.model_dump())
            return ("metric", m)
        return None

    def _alarm(self, code: str, severity: str, title: str, detail: str, turn: int):
        alarm = dispatch(Alarm(code=code, severity=severity, title=title, detail=detail, flight=self.s.flight, turn=turn))
        self.alarms.append(alarm)
        self.log.add(turn, "SYSTEM", "alarm", alarm.model_dump())
        return ("alarm", alarm)

    def best_partial(self) -> Optional[tuple[Proposal, float]]:
        """Best fully-safe plan any participant has put forward so far ("keep what we have")."""
        best = None
        for p in self.seen_plans.values():
            r = check_all(self.s, p)
            if is_safe(r):
                c = cost(self.s, p, r)
                if best is None or c < best[1]:
                    best = (p, c)
        return best

    def decision_support(self) -> str:
        """USD price and safety of every plan seen so far, for the Chief of Staff (no private raw data)."""
        rows = []
        for p in list(self.seen_plans.values())[-8:]:
            r = check_all(self.s, p)
            if is_safe(r):
                items = sorted(cost_breakdown(self.s, p, r).items(), key=lambda kv: -kv[1])[:2]
                rows.append(f"- {self._describe(p)}: SAFE, ${cost(self.s, p, r):,.0f} "
                            f"(largest: {', '.join(f'{k} ${v:,.0f}' for k, v in items if v)})")
            else:
                rows.append(f"- {self._describe(p)}: UNSAFE ({'; '.join(c.name for x in r.values() for c in x.violations())})")
        return "\n".join(rows)

    def _describe(self, p: Proposal) -> str:
        return f"TLDT {self.s.clock(p.landing_slot_min)} (slot {p.landing_slot_min}), RWY {p.runway}, stand {p.gate}"

    def _decide(self, req: DecisionRequest) -> Iterator[Event]:
        """Ask a human (or take the default in AUTONOMOUS). Generator: returns the chosen option id."""
        self.log.add(req.turn, "SYSTEM", "decision_request", req.model_dump())
        if self.autonomy == Autonomy.AUTONOMOUS:
            choice, by = req.default, "AUTO"
            yield ("decision_auto", {"request": req, "choice": choice})
        else:
            choice = yield ("decision", req)
            by = "HUMAN"
            if choice not in {o.id for o in req.options}:
                choice, by = req.default, "AUTO"
        rec = DecisionRecord(code=req.code, turn=req.turn, option=choice, by=by)
        self.decisions.append(rec)
        self.log.add(req.turn, by, "decision", rec.model_dump())
        return choice

    def _exit_options(self, include_retry: Optional[str] = None) -> list[DecisionOption]:
        opts = []
        if include_retry:
            opts.append(DecisionOption(id="retry", label="Wait and retry", description=include_retry))
        partial = self.best_partial()
        if partial:
            opts.append(DecisionOption(id="keep_partial", label="Stop and keep the best safe plan so far",
                                       description=f"{self._describe(partial[0])}, cost {partial[1]}. Already passes "
                                                   "every hard check; not all agents have signed it.",
                                       proposal=partial[0]))
        ref = self.reference
        opts.append(DecisionOption(id="fallback", label="Hand over to the deterministic planner",
                                   description=f"{ref.mode}: {self._describe(ref.proposal)}"
                                               + (f", cost {ref.cost}" if ref.cost is not None else ""),
                                   proposal=ref.proposal))
        return opts

    def _ask_agents(self, turn: int, calls: dict[str, Callable[[], object]]) -> Iterator[Event]:
        """Run agent calls (in parallel when >1). On failure or timeout: alarm, then a decision to wait and
        retry, keep the best safe plan so far, or hand over to the planner. Returns {agent: message}."""
        results: dict[str, object] = {}
        pending = dict(calls)
        attempts = 0
        while pending:
            yield ("thinking", {"agents": list(pending), "turn": turn})
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                futures = {k: pool.submit(fn) for k, fn in pending.items()}
                errors: dict[str, AgentError] = {}
                for k, f in futures.items():
                    try:
                        results[k] = f.result()
                        ev = self._metric(getattr(results[k], "metric", None))
                        if ev:
                            yield ev
                    except AgentError as e:
                        errors[k] = e
                        ev = self._metric(e.metric)
                        if ev:
                            yield ev
            pending = {k: v for k, v in pending.items() if k in errors}
            if not errors:
                break
            timeout = any(e.timeout for e in errors.values())
            detail = "; ".join(str(e) for e in errors.values())
            yield self._alarm("AGENT_TIMEOUT" if timeout else "AGENT_FAILURE", "ALARM",
                              "Agent call timed out" if timeout else "Agent call failed", detail, turn)
            for k in errors:  # give a retry twice the time
                self.agents[k].timeout_s *= 2
            req = DecisionRequest(
                code="AGENT_TIMEOUT" if timeout else "AGENT_FAILURE", severity="ALARM", turn=turn,
                title=f"{', '.join(errors)} did not answer",
                briefing=[detail, f"Nothing is assumed to have succeeded: no plan is committed. {len(self.messages)} "
                          "messages and every safe plan seen so far are preserved."],
                options=self._exit_options(include_retry=f"Retry {', '.join(errors)} with a "
                                                         f"{max(self.agents[k].timeout_s for k in errors):.0f}s timeout."),
                default="retry" if attempts == 0 else ("keep_partial" if self.best_partial() else "fallback"))
            choice = yield from self._decide(req)
            attempts += 1
            if choice != "retry":
                raise _Stop(choice)
        return results

    # ------------------------------------------------------------------ finishing
    def _finish(self, status, source: str, turns: int, proposal: Proposal, reason: Optional[str] = None) -> Event:
        s = self.s
        reports = check_all(s, proposal) if s.runway(proposal.runway) else {}
        final_cost = cost(s, proposal, reports) if reports and is_safe(reports) else None
        self.log.add(turns, "SYSTEM", "decision_final", {
            "status": status, "plan_source": source, "proposal": proposal.model_dump(), "cost": final_cost,
            "reference_cost": self.reference.cost, "reason": reason})
        path = None
        if self.save_log:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = str(self.log.save(LOG_DIR / f"{s.flight}_{s.date}_{stamp}.jsonl"))
        self.outcome = Outcome(
            status=status, plan_source=source, turns_used=turns, proposal=proposal, reports=reports, cost=final_cost,
            reference=self.reference, reason=reason, messages=self.messages, briefs=self.briefs,
            decisions=self.decisions, alarms=self.alarms, metrics=self.metrics, usage=summarise(self.metrics),
            deadlock=self.deadlock, deadline_hit=self.deadline_hit, safety_overrides=self.safety_overrides, stale_replies=self.stale_replies,
            log_path=path, log_verified=self.log.verify(), elapsed_s=round(time.monotonic() - self.started, 1))
        return ("outcome", self.outcome)

    def _fallback(self, turn: int, why: str) -> Iterator[Event]:
        ref = self.reference
        yield self._alarm("FALLBACK", "ALARM" if ref.mode != "OPTIMISED_SOP" else "WARNING",
                          f"Deterministic planner engaged ({ref.mode})", why, turn)
        options = [DecisionOption(id="approve", label=f"Approve planner plan ({ref.mode})",
                                  description=f"{self._describe(ref.proposal)}. {ref.reason}", proposal=ref.proposal)]
        options += [DecisionOption(id=f"alt{i}", label=f"Use safe alternative {i}",
                                   description=f"{self._describe(p)}, cost {c}", proposal=p)
                    for i, (p, c) in enumerate(ref.alternatives, 1)]
        partial = self.best_partial()
        if partial and _key(partial[0]) != _key(ref.proposal):
            options.append(DecisionOption(id="keep_partial", label="Use the best safe plan the agents produced",
                                          description=f"{self._describe(partial[0])}, cost {partial[1]}",
                                          proposal=partial[0]))
        req = DecisionRequest(code="FALLBACK_REVIEW", severity="WARNING", turn=turn,
                              title="No agreed plan: confirm the non-AI plan", briefing=[why, ref.reason],
                              options=options, default="approve")
        choice = yield from self._decide(req)
        yield self._note(turn, "fallback", f"{why} Planner: {ref.mode} {self._describe(ref.proposal)}.")
        chosen = next(o for o in options if o.id == choice)
        if choice == "approve":
            return self._finish("FALLBACK", f"deterministic planner ({ref.mode})", turn, ref.proposal, f"{why} {ref.reason}")
        return self._finish("HUMAN_DECISION", f"human selected {chosen.label.lower()}", turn, chosen.proposal, why)

    def _stop(self, turn: int, stop: _Stop) -> Iterator[Event]:
        if stop.choice == "keep_partial" and self.best_partial():
            p, _ = self.best_partial()
            return self._finish("HUMAN_DECISION", "best safe plan so far (kept on human/auto decision)", turn, p,
                                "Negotiation stopped early; partial result kept.")
        if stop.choice == "plan" and stop.plan is not None:
            return self._finish("HUMAN_DECISION", "human-approved plan", turn, stop.plan)
        return (yield from self._fallback(turn, "Negotiation stopped before consensus."))

    # ------------------------------------------------------------------ main loop
    def _run(self) -> Iterator[Event]:
        final = yield from self._inner()
        yield final

    def _inner(self) -> Iterator[Event]:
        s = self.s
        self.log.add(0, "SYSTEM", "scenario", {
            "flight": s.flight, "route": f"{s.origin}-{s.destination}", "date": s.date, "dot_row": s.row_index,
            "weather_delay_min": s.weather_delay_min, "late_aircraft_delay_min": s.late_aircraft_delay_min,
            "weather_mode": s.weather_mode, "weather_attribution": s.weather_attribution,
            "weather_sources": [x.model_dump() for x in (s.weather.sources if s.weather else [])],
            "autonomy": self.autonomy.value, "max_turns": MAX_TURNS,
            "models": {k: a.model for k, a in self.agents.items()}})
        turn = 0
        try:
            yield from self._moderate(0, opening=True)
            on_table: Optional[Proposal] = None
            plan_id = 0
            last_countered: Optional[tuple] = None
            for turn in range(1, MAX_TURNS + 1):
                if self._over_budget():
                    yield self._alarm("TIME_BUDGET", "ALARM", "Negotiation time budget exceeded",
                                      f"{time.monotonic() - self.started:.1f}s > {self.deadline_s:.0f}s budget", turn)
                    raise _Stop("keep_partial" if self.best_partial() else "fallback")
                if turn > 1:
                    yield from self._moderate(turn - 1, opening=False)
                yield ("turn", {"turn": turn})

                # 1. ATC tables a plan --------------------------------------------------
                prev = VALIDATORS["ATC"](s, on_table) if on_table else None
                got = yield from self._ask_agents(turn, {"ATC": lambda: self.agents["ATC"].act(
                    s, turn, self._bus(turn), on_table, plan_id, prev)})
                atc_msg: AgentMessage = got["ATC"]
                plan_id += 1
                atc_msg.stance, atc_msg.plan_id, atc_msg.answering_plan = "PROPOSE", plan_id, 0
                yield self._record(atc_msg)
                proposal = atc_msg.proposal

                atc_report = VALIDATORS["ATC"](s, proposal)
                if not atc_report.hard_ok:
                    yield self._note(turn, "safety_block", f"Plan #{plan_id} blocked by the ATC validator: "
                                     + "; ".join(c.detail for c in atc_report.violations()))
                    continue
                if last_countered == _key(proposal):
                    self.deadlock = True
                    yield self._alarm("DEADLOCK", "ALARM", "Negotiation deadlock",
                                      f"ATC re-tabled plan {self._describe(proposal)} unchanged after it was countered.",
                                      turn)
                    opts = self._exit_options()
                    if turn < MAX_TURNS:
                        opts.insert(0, DecisionOption(id="one_more_turn", label="Allow one more turn",
                                                      description="Let the specialists try once more."))
                    req = DecisionRequest(code="DEADLOCK", severity="ALARM", turn=turn, title="Specialists are stuck",
                                          briefing=[f"Plan #{plan_id} repeats a plan that was already countered.",
                                                    "Continuing unchanged would burn turns without progress."],
                                          options=opts, default="keep_partial" if self.best_partial() else "fallback")
                    choice = yield from self._decide(req)
                    if choice != "one_more_turn":
                        raise _Stop(choice)
                on_table = proposal

                # 2-3. Flight Crew + AOCC answer in parallel ----------------------------
                own = {d: VALIDATORS[d](s, proposal) for d in ("PILOT", "GROUND")}
                pid = plan_id
                replies = yield from self._ask_agents(turn, {
                    d: (lambda d=d: self.agents[d].act(s, turn, self._bus(turn), proposal, pid, own[d]))
                    for d in ("PILOT", "GROUND")})
                accepts = 0
                for d in ("PILOT", "GROUND"):
                    msg: AgentMessage = replies[d]
                    if msg.stance == "PROPOSE" or (msg.stance == "ACCEPT" and _key(msg.proposal) != _key(proposal)):
                        msg.stance = "COUNTER"  # "accepting" a different plan is a counter
                    yield self._record(msg)
                    if msg.answering_plan != plan_id:
                        self.stale_replies += 1
                        yield self._note(turn, "stale_reply", f"{d} answered plan #{msg.answering_plan}, current is "
                                         f"#{plan_id}: reply discarded.")
                        continue
                    if msg.stance == "ACCEPT":
                        if own[d].hard_ok:
                            accepts += 1
                        else:
                            self.safety_overrides += 1
                            yield self._note(turn, "safety_override", f"{d} ACCEPT overridden: plan #{plan_id} breaks "
                                             + "; ".join(c.detail for c in own[d].violations()))
                            yield self._alarm("SAFETY_OVERRIDE", "WARNING", f"{d} accepted an unsafe plan",
                                              "Overridden by the deterministic validator.", turn)
                    elif not VALIDATORS[d](s, msg.proposal).hard_ok:
                        yield self._note(turn, "invalid_counter", f"{d} counter breaks its own hard limits.")
                if accepts < 2:
                    last_countered = _key(proposal)

                # 4. Consensus check ---------------------------------------------------------
                full = check_all(s, proposal)
                self.log.add(turn, "SYSTEM", "validation", {
                    "plan_id": plan_id, "proposal": proposal.model_dump(),
                    **{d: {"hard_ok": r.hard_ok, "violations": [c.detail for c in r.violations()],
                           "warnings": [c.detail for c in r.warnings()], "metrics": r.metrics} for d, r in full.items()}})
                yield ("validation", {"turn": turn, "plan_id": plan_id, "reports": full})
                if accepts == 2 and is_safe(full):
                    risks = self._risks(full)
                    if self.autonomy == Autonomy.STEP_BY_STEP or (self.autonomy == Autonomy.SUPERVISED and risks):
                        opts = [DecisionOption(id="approve", label=f"Approve plan #{plan_id}",
                                               description=self._describe(proposal), proposal=proposal)]
                        if turn < MAX_TURNS:
                            opts.append(DecisionOption(id="continue", label="Reject, keep negotiating",
                                                       description="Send the specialists back for another turn."))
                        opts += [o for o in self._exit_options() if o.id == "fallback"]
                        req = DecisionRequest(code="CONSENSUS_REVIEW", severity="WARNING" if risks else "INFO",
                                              turn=turn, title=f"Specialists agree on plan #{plan_id}",
                                              briefing=[self._describe(proposal)] + (risks or ["No risk flags."]),
                                              options=opts, default="approve")
                        choice = yield from self._decide(req)
                        if choice == "continue":
                            yield self._note(turn, "human_rejected", f"Plan #{plan_id} rejected by the human "
                                             "supervisor; renegotiate.")
                            last_countered = _key(proposal)
                            continue
                        if choice == "fallback":
                            raise _Stop("fallback")
                    yield self._note(turn, "consensus", f"Plan #{plan_id} agreed by all three specialists and passes "
                                     f"every hard check (turn {turn}/{MAX_TURNS}).")
                    return self._finish("CONSENSUS", "agent consensus", turn, proposal)
                if accepts == 2:
                    yield self._note(turn, "safety_block", "Unanimous ACCEPT but cross-domain validation failed: "
                                     + "; ".join(c.detail for r in full.values() for c in r.violations()))

                if self.autonomy == Autonomy.STEP_BY_STEP and turn < MAX_TURNS:
                    req = DecisionRequest(code="TURN_REVIEW", severity="INFO", turn=turn,
                                          title=f"Turn {turn} ended without agreement",
                                          briefing=[f"Plan #{plan_id}: {self._describe(proposal)}",
                                                    *(render_message(m, s).splitlines()[0] for m in self.messages
                                                      if m.turn == turn)],
                                          options=[DecisionOption(id="continue", label="Continue to the next turn",
                                                                  description="Chief of Staff briefs, ATC re-tables.")]
                                          + self._exit_options(), default="continue")
                    choice = yield from self._decide(req)
                    if choice != "continue":
                        raise _Stop(choice)
        except _Stop as stop:
            return (yield from self._stop(max(turn, 1), stop))
        return (yield from self._fallback(MAX_TURNS, f"No safe consensus within {MAX_TURNS} turns."))

    def _over_budget(self) -> bool:
        if self.deadline_s is not None and time.monotonic() - self.started > self.deadline_s:
            self.deadline_hit = True
        return self.deadline_hit

    def _risks(self, full: dict[Domain, DomainReport]) -> list[str]:
        pm, gm = full["PILOT"].metrics, full["GROUND"].metrics
        risks = []
        if pm.get("fuel_margin_min", 99) < ops_settings.get().min_fuel_advisory_margin_min:
            risks.append(f"Fuel margin {pm['fuel_margin_min']} min above FRF+ALTN: crew will declare MINIMUM FUEL.")
        if pm.get("holding_min", 0) > 0 and "MOD" in (pm.get("hold_turbulence"), pm.get("hold_icing")):
            risks.append(f"{pm['holding_min']} min holding in reported MOD turbulence/icing.")
        if gm.get("crew_overtime_min", 0) > 0:
            risks.append(f"Ground crew overtime {gm['crew_overtime_min']} min.")
        if pm.get("missed_connection_pax", 0) > 0:
            risks.append(f"{pm['missed_connection_pax']} connecting passengers miss onward flights.")
        if gm.get("next_leg_delay_min", 0) > 15:
            risks.append(f"Next leg knock-on delay {gm['next_leg_delay_min']} min.")
        return risks

    def _moderate(self, turn: int, opening: bool) -> Iterator[Event]:
        mod = self.agents.get("MODERATOR")
        if not isinstance(mod, ModeratorAgent):
            return
        yield ("thinking", {"agents": ["MODERATOR"], "turn": turn})
        try:
            brief = mod.brief(self.s, turn, self._bus(turn + 1), opening, support=self.decision_support())
        except AgentError as e:  # advisory role: never blocks the negotiation
            ev = self._metric(e.metric)
            if ev:
                yield ev
            yield self._note(turn, "moderator_unavailable", f"Chief of Staff unavailable ({e}); continuing.")
            return
        ev = self._metric(brief.metric)
        if ev:
            yield ev
        self.briefs.append(brief)
        self.public.append((turn + 1, render_brief(brief)))
        if brief.bridge_proposal:
            self.seen_plans.setdefault(_key(brief.bridge_proposal), brief.bridge_proposal)
        payload = brief.model_dump(exclude={"metric"})
        if brief.bridge_proposal:  # audit only; agents are not told whether it is safe
            payload["bridge_proposal_safe"] = is_safe(check_all(self.s, brief.bridge_proposal))
        self.log.add(turn, "MODERATOR", "brief", payload)
        yield ("brief", brief)


def negotiate(s: Scenario, agents: dict[str, BaseAgent], on_event: Optional[Callable[[str, object], None]] = None,
              autonomy: Autonomy = Autonomy.AUTONOMOUS, decide: Optional[Callable[[DecisionRequest], str]] = None,
              save_log: bool = True, deadline_s: Optional[float] = None) -> Outcome:
    """Run a session to completion. `decide` answers decision requests (default: the safe default)."""
    session = NegotiationSession(s, agents, autonomy, save_log, deadline_s)
    kind, data = session.advance()
    while True:
        if on_event:
            on_event(kind, data)
        if kind == "outcome":
            return data  # type: ignore[return-value]
        if kind == "decision":
            kind, data = session.advance(decide(data) if decide else data.default)
        else:
            kind, data = session.advance()
