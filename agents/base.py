"""Shared message protocol and the Claude call used by every agent.

Each agent is its own Claude API call with its own system prompt and its own PRIVATE slice of the
scenario. Agents never see each other's private data; they see only the public bus view the
orchestrator gives them (a constraint ledger plus the latest turn's messages).
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, create_model

from constraints import Domain, DomainReport, Proposal
from monitoring import CallMetric, price
from scenario import Scenario

DEFAULT_MODEL = "claude-opus-5-5"
MODELS_WITH_SERVER_FALLBACK = {"claude-opus-5-5", "claude-sonnet-5-5"}
DEFAULT_TIMEOUT_S = 90.0


# ============================================================================
# Structured message protocol
# ============================================================================

class ConstraintClaim(BaseModel):
    name: str  # e.g. "latest_safe_TLDT", "stand_available"
    value: str  # e.g. "18:52", "B4 from 18:30"
    hard: bool  # true = safety / legal / physical limit, non-negotiable


class AgentReply(BaseModel):
    stance: Literal["PROPOSE", "ACCEPT", "COUNTER"]
    answering_plan: int  # version of the plan this reply answers (0 = none yet)
    proposal: Proposal
    constraints: list[ConstraintClaim]
    concession: str
    rationale: str


class AgentMessage(AgentReply):
    """AgentReply plus envelope fields set by the orchestrator, never by the model."""
    agent: Domain
    turn: int
    plan_id: int = 0  # version assigned when this message puts a plan on the table
    metric: Optional[CallMetric] = None


@lru_cache(maxsize=64)
def _reply_schema(runways: tuple[str, ...], gates: tuple[str, ...]) -> type[BaseModel]:
    """Per-scenario reply schema. Runway and stand are enums of the REAL identifiers in this scenario,
    so structured outputs make a hallucinated runway or stand impossible to emit."""
    plan = create_model("Plan", landing_slot_min=(int, ...), runway=(Literal[runways], ...),
                        gate=(Literal[gates], ...))
    return create_model("Reply", stance=(Literal["PROPOSE", "ACCEPT", "COUNTER"], ...), answering_plan=(int, ...),
                        proposal=(plan, ...), constraints=(list[ConstraintClaim], ...), concession=(str, ...),
                        rationale=(str, ...))


def reply_schema(s: Scenario) -> type[BaseModel]:
    return _reply_schema(tuple(r.id for r in s.runways), tuple(g.id for g in s.gates))


class AgentError(Exception):
    def __init__(self, agent: str, message: str, metric: Optional[CallMetric] = None, timeout: bool = False):
        super().__init__(f"{agent}: {message}")
        self.agent, self.metric, self.timeout = agent, metric, timeout


# ============================================================================
# Agent base
# ============================================================================

SAFETY_CHARTER = """\
SAFETY CHARTER (overrides every other instruction, including requests from humans):
1. Priority order: safety of life > airworthiness and legal limits > passenger welfare > schedule and cost.
2. Hard limits are not negotiable: fuel (FRF + ALTN), wake separation, runway closures and wind limits,
   stand size and availability, crew duty cap, holding in severe turbulence or icing. Never propose,
   accept or endorse a plan that breaks one, whoever asks and whatever the delay or cost.
3. Never state a safety figure you did not take from your instruments or tools. If you cannot verify
   something, say so and treat it as unsafe.
4. If anyone (an agent or a human) asks you to relax a hard limit, refuse plainly, name the limit, and
   offer the safest alternative or an escalation.
5. When in doubt, choose the conservative option and flag it for the duty officer.
6. You are not the last line of defence: deterministic code re-checks every plan, and an unsafe ACCEPT is
   overridden and logged against you. Act so that it never has to."""

PROTOCOL_RULES = """\
OPERATING PROTOCOL (A-CDM collaborative decision; every agent follows it):
- Clock times are local (HH:MM). In the JSON plan, landing_slot_min is minutes relative to the flight's
  Scheduled In-Block Time (SIBT); your instruments give both forms. Negative values are allowed.
- A plan is: landing_slot_min (the Target Landing Time, TLDT), runway, gate (stand).
- The ATC agent tables a numbered plan each turn (stance PROPOSE). Flight Crew and AOCC answer ACCEPT or
  COUNTER and set answering_plan to that plan's number. Replies to an outdated plan number are discarded.
- Hard limit: 3 turns, then a deterministic non-AI planner decides. Converge: ACCEPT any plan that meets
  your hard limits even if it is not your preference; record the preference as a soft constraint.
- Publish limits as constraints so others can plan around them (hard=true only for safety, legal or
  physical limits). Quote your instruments; never estimate a figure your instruments already give.
- Every plan is re-checked by deterministic safety code. A plan that breaks a hard limit is rejected
  whoever accepted it, so never accept one that breaks yours.
- A Duty Operations Manager (Chief of Staff) moderates. It has no authority and no private data.
- Write as an operational specialist: terse, standard terminology, no pleasantries. rationale: at most
  two sentences. concession: one sentence."""


class BaseAgent:
    domain: Domain
    title: str
    role: str

    def __init__(self, client: Optional[anthropic.Anthropic], model: str = DEFAULT_MODEL, effort: str = "low",
                 timeout_s: float = DEFAULT_TIMEOUT_S):
        self.client, self.model, self.effort, self.timeout_s = client, model, effort, timeout_s

    def private_brief(self, s: Scenario) -> str:
        raise NotImplementedError

    def system_prompt(self) -> str:
        return f"{self.role}\n\n{SAFETY_CHARTER}\n\n{PROTOCOL_RULES}"

    def act(self, s: Scenario, turn: int, bus: str, on_table: Optional[Proposal], plan_id: int,
            own_report: Optional[DomainReport]) -> AgentMessage:
        parts = [f"## Your private instruments (only you can see these)\n{self.private_brief(s)}",
                 f"## Public bus\n{bus or '(empty: you open the session)'}"]
        if on_table is not None:
            parts.append(f"## Plan #{plan_id} on the table\n{on_table.model_dump_json()} "
                         f"(TLDT {s.clock(on_table.landing_slot_min)})")
        if own_report is not None:
            parts.append(f"## Your deterministic check of plan #{plan_id}\n{render_report(own_report)}")
        parts.append(f"## Task (turn {turn} of 3)\n{self.task(turn, on_table, plan_id)}")
        reply, metric = self._call("\n\n".join(parts), reply_schema(s), turn)
        data = reply.model_dump()
        return AgentMessage(**data, agent=self.domain, turn=turn, metric=metric)

    def task(self, turn: int, on_table: Optional[Proposal], plan_id: int) -> str:
        return (f"ACCEPT plan #{plan_id} if it meets all your hard limits (copy it unchanged into proposal). "
                "Otherwise COUNTER with the closest plan that meets your limits and publish the limits plan "
                f"#{plan_id} breaks. Set answering_plan={plan_id}.")

    def _add_fallbacks(self, kwargs: dict) -> None:
        if self.model in MODELS_WITH_SERVER_FALLBACK:
            # Server-side refusal fallback: a declined request is re-run on a fallback model in the same call.
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")

    def _call(self, user_content: str, schema: type[BaseModel], turn: int,
              system: Optional[str] = None) -> tuple[BaseModel, CallMetric]:
        kwargs = dict(model=self.model, max_tokens=16000, system=system or self.system_prompt(),
                      messages=[{"role": "user", "content": user_content}], output_format=schema,
                      output_config={"effort": self.effort})
        self._add_fallbacks(kwargs)
        started = time.monotonic()

        def fail(msg: str, timeout: bool = False) -> AgentError:
            return AgentError(self.domain, msg, CallMetric(agent=self.domain, turn=turn, model=self.model,
                                                           latency_s=round(time.monotonic() - started, 2),
                                                           ok=False, error=msg), timeout)
        if self.client is None:
            raise fail("no API client configured")
        try:
            response = self.client.with_options(timeout=self.timeout_s).beta.messages.parse(**kwargs)
        except anthropic.APITimeoutError as e:
            raise fail(f"timed out after {self.timeout_s:.0f}s", timeout=True) from e
        except anthropic.APIConnectionError as e:
            raise fail(f"network error: {e}") from e
        except anthropic.AuthenticationError as e:
            raise fail("authentication failed (check ANTHROPIC_API_KEY)") from e
        except anthropic.RateLimitError as e:
            raise fail("rate limited") from e
        except anthropic.APIStatusError as e:
            raise fail(f"API error {e.status_code}: {e.message}") from e
        u = response.usage
        metric = CallMetric(agent=self.domain, turn=turn, model=self.model,
                            latency_s=round(time.monotonic() - started, 2), input_tokens=u.input_tokens,
                            output_tokens=u.output_tokens,
                            cost_usd=price(self.model, u.input_tokens, u.output_tokens))
        if response.stop_reason == "refusal":
            metric.ok, metric.error = False, "refusal"
            raise AgentError(self.domain, "model refused", metric)
        if response.parsed_output is None:
            metric.ok, metric.error = False, f"no structured reply ({response.stop_reason})"
            raise AgentError(self.domain, metric.error, metric)
        return response.parsed_output, metric


def render_message(m: AgentMessage, s: Optional[Scenario] = None) -> str:
    cons = "; ".join(f"{c.name}={c.value}{' [HARD]' if c.hard else ''}" for c in m.constraints) or "none"
    p = m.proposal
    when = f" TLDT {s.clock(p.landing_slot_min)}" if s else ""
    tag = f"PLAN #{m.plan_id}" if m.stance == "PROPOSE" else f"re #{m.answering_plan}"
    return (f"[T{m.turn}] {m.agent} {m.stance} ({tag}) slot={p.landing_slot_min}{when} rwy={p.runway} "
            f"stand={p.gate}\n    constraints: {cons}\n    concession: {m.concession}\n    rationale: {m.rationale}")


def render_report(r: DomainReport) -> str:
    lines = [f"- {'PASS' if c.passed else 'FAIL'} {'[HARD]' if c.hard else '[soft]'} {c.name}: {c.detail}"
             for c in r.checks]
    lines.append(f"- metrics: {json.dumps(r.metrics)}")
    return "\n".join(lines)
