"""Chief of Staff: the Duty Operations Manager who runs the meeting but never decides it.

It sees ONLY public information (the DOT disruption facts, the public weather summary and the bus), none
of the domain agents' private instruments. It opens the session, and after each turn without consensus it
names the open conflicts and may suggest a bridge plan. It has no vote: consensus still needs all three
domain agents plus the deterministic safety checks.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, create_model

from agents.base import BaseAgent, ConstraintClaim, reply_schema  # noqa: F401
from constraints import Proposal
from monitoring import CallMetric
from scenario import Scenario


class Directive(BaseModel):
    agent: Literal["ATC", "PILOT", "GROUND"]
    instruction: str


class ModeratorReply(BaseModel):
    summary: str
    open_conflicts: list[str]
    bridge_proposal: Optional[Proposal]
    directives: list[Directive]


class ModeratorMessage(ModeratorReply):
    agent: Literal["MODERATOR"] = "MODERATOR"
    turn: int
    metric: Optional[CallMetric] = None


def _moderator_schema(s: Scenario) -> type[BaseModel]:
    plan = reply_schema(s).model_fields["proposal"].annotation  # same real-ID enums as the agents
    return create_model("Brief", summary=(str, ...), open_conflicts=(list[str], ...),
                        bridge_proposal=(Optional[plan], ...), directives=(list[Directive], ...))


class ModeratorAgent(BaseAgent):
    domain = "MODERATOR"  # type: ignore[assignment]
    title = "Chief of Staff: Duty Operations Manager"
    role = ("You are the Duty Operations Manager chairing an A-CDM disruption call between three independent "
            "specialists: ATC (arrival manager), Flight Crew (captain) and AOCC (turnaround coordinator). You "
            "have NO operational authority and NO private data: only the public disruption facts and the "
            "public bus. Make them converge safely inside the turn limit: name each conflict precisely, and "
            "when the published constraints allow it, propose one bridge plan that satisfies every HARD "
            "constraint any specialist has published. Never relax or argue against a hard constraint. Be "
            "terse: summary at most two sentences, one sentence per directive.")

    def public_brief(self, s: Scenario) -> str:
        return (f"Flight {s.flight} ({s.airline}, tail {s.tail_number}, {s.aircraft_type}) {s.origin}-{s.destination} "
                f"({s.destination_name}) on {s.date}, SIBT {s.clock(0)} local. US DOT 2015 record: WEATHER_DELAY "
                f"{s.weather_delay_min} min, LATE_AIRCRAFT_DELAY {s.late_aircraft_delay_min} min. "
                f"Weather: {s.weather_attribution} Runways in use: {', '.join(r.id for r in s.runways)}.")

    def brief(self, s: Scenario, turn: int, bus: str, opening: bool) -> ModeratorMessage:
        task = ("Open the call: summarise the disruption and tell each specialist what to publish first so the "
                "call can close in as few turns as possible. bridge_proposal must be null."
                if opening else
                f"Turn {turn} ended without consensus; {3 - turn} turn(s) remain before the deterministic planner "
                "takes over. Summarise, list open conflicts, propose a bridge plan if the published constraints "
                "allow one, and give each specialist a directive for the next turn.")
        content = (f"## Public disruption facts\n{self.public_brief(s)}\n\n## Public bus\n{bus or '(empty)'}"
                   f"\n\n## Task\n{task}")
        reply, metric = self._call(content, _moderator_schema(s), turn, system=self.role)
        data = reply.model_dump()
        return ModeratorMessage(**data, turn=turn, metric=metric)


def render_brief(m: ModeratorMessage) -> str:
    lines = [f"[T{m.turn}] CHIEF OF STAFF: {m.summary}"]
    if m.open_conflicts:
        lines.append("    open conflicts: " + "; ".join(m.open_conflicts))
    if m.bridge_proposal:
        bp = m.bridge_proposal
        lines.append(f"    suggested bridge plan: slot={bp.landing_slot_min} rwy={bp.runway} stand={bp.gate}")
    lines += [f"    -> {d.agent}: {d.instruction}" for d in m.directives]
    return "\n".join(lines)
