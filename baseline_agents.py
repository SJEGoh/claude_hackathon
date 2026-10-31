"""Rule-based (non-LLM) specialists with the same interface as the Claude agents.

Uses: a baseline to compare the AI agents against in batch evaluation, an offline demo mode, and the
test suite (no API calls, no cost). Each one only reads its own domain's validator, like the AI agents.
"""
from __future__ import annotations

from typing import Optional

from agents.base import AgentMessage, ConstraintClaim, StakeholderVote
from constraints import Domain, DomainReport, Proposal, check_atc, check_ground, check_pilot
from scenario import Scenario


class _Rule:
    model = "rule-based"
    timeout_s = 0.0

    def __init__(self, domain: Domain):
        self.domain = domain

    def _msg(self, stance, turn, plan_ref, p: Proposal, report: Optional[DomainReport], why: str) -> AgentMessage:
        claims = [ConstraintClaim(name=c.name, value=c.detail[:80], hard=c.hard)
                  for c in (report.checks if report else []) if not c.passed]
        return AgentMessage(stance=stance, answering_plan=plan_ref, proposal=p, constraints=claims,
                            concession="-", rationale=why, agent=self.domain, turn=turn)

    def ratify(self, s: Scenario, turn: int, plan_id: int, briefing: str, own_report: DomainReport) -> StakeholderVote:
        why = "Within my limits." if own_report.hard_ok else "; ".join(c.detail for c in own_report.violations())
        return StakeholderVote(agree=own_report.hard_ok, reason=why, agent=self.domain, turn=turn)


class RuleATC(_Rule):
    def __init__(self):
        super().__init__("ATC")
        self.counters: list[Proposal] = []

    def act(self, s: Scenario, turn: int, bus: str, on_table, plan_id, own_report) -> AgentMessage:
        # Adopt the most recent counter that ATC can live with; otherwise the earliest free slot.
        for p in reversed(self.counters):
            if check_atc(s, p).hard_ok and (on_table is None or p != on_table):
                return self._msg("PROPOSE", turn, 0, p, None, "Adopting counter-proposal.")
        start = s.eta_min if on_table is None else on_table.landing_slot_min + 1
        for t in range(start, start + 180):
            for r in s.runways:
                p = Proposal(landing_slot_min=t, runway=r.id, gate=on_table.gate if on_table else s.scheduled_gate)
                if check_atc(s, p).hard_ok:
                    return self._msg("PROPOSE", turn, 0, p, None, "Earliest conflict-free slot.")
        return self._msg("PROPOSE", turn, 0, on_table or Proposal(landing_slot_min=s.eta_min, runway=s.runways[0].id,
                                                                  gate=s.scheduled_gate), None, "No free slot.")


class RuleResponder(_Rule):
    def __init__(self, domain: Domain, atc: RuleATC):
        super().__init__(domain)
        self.atc = atc

    def act(self, s: Scenario, turn: int, bus: str, on_table: Proposal, plan_id: int, own_report) -> AgentMessage:
        check = check_pilot if self.domain == "PILOT" else check_ground
        if own_report.hard_ok:
            return self._msg("ACCEPT", turn, plan_id, on_table, own_report, "Within my limits.")
        gates = [on_table.gate] if self.domain == "PILOT" else [g.id for g in s.gates]
        for t in range(on_table.landing_slot_min, on_table.landing_slot_min + 120):
            for gid in gates:
                p = Proposal(landing_slot_min=t, runway=on_table.runway, gate=gid)
                if check(s, p).hard_ok and check_atc(s, p).hard_ok:
                    self.atc.counters.append(p)
                    return self._msg("COUNTER", turn, plan_id, p, own_report, "Closest plan inside my limits.")
            if self.domain == "PILOT":  # pilot never asks for a later slot than needed
                break
        return self._msg("COUNTER", turn, plan_id, on_table, own_report, "No acceptable alternative found.")


def make_rule_agents() -> dict:
    atc = RuleATC()
    return {"ATC": atc, "PILOT": RuleResponder("PILOT", atc), "GROUND": RuleResponder("GROUND", atc)}


# ============================================================================
# Safety drill: adversarial agents that try to push an unsafe plan through
# ============================================================================

class RogueATC(_Rule):
    """Tables plans that keep wake separation but land after the aircraft's fuel limit."""

    def __init__(self):
        super().__init__("ATC")

    def act(self, s: Scenario, turn: int, bus: str, on_table, plan_id, own_report) -> AgentMessage:
        start = s.latest_safe_landing + 3 + turn  # deliberately past the protected reserves, a new plan each turn
        for t in range(start, start + 120):
            p = Proposal(landing_slot_min=t, runway=s.runways[0].id, gate="R1")
            if check_atc(s, p).hard_ok:
                return self._msg("PROPOSE", turn, 0, p, None,
                                 "DRILL: maximise runway throughput; fuel is the crew's problem.")
        return self._msg("PROPOSE", turn, 0, Proposal(landing_slot_min=start, runway=s.runways[0].id, gate="R1"),
                         None, "DRILL")


class YesMan(_Rule):
    """Accepts whatever is on the table: the failure mode the safety engine must catch."""

    def act(self, s: Scenario, turn: int, bus: str, on_table: Proposal, plan_id: int, own_report) -> AgentMessage:
        return self._msg("ACCEPT", turn, plan_id, on_table, None, "DRILL: accepting to avoid delay.")


def make_drill_agents() -> dict:
    return {"ATC": RogueATC(), "PILOT": YesMan("PILOT"), "GROUND": YesMan("GROUND")}
