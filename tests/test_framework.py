"""Offline tests: data, weather parsing, validators, planner, negotiation engine, HITL gates, escalation, log.

Agents are scripted stubs and scenarios use weather_mode="OFF", so no API key, credits or network needed.
Run: .venv/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os  # noqa: E402
os.environ["OPS_SETTINGS_FILE"] = "/nonexistent/ops_settings.json"  # tests run on default limits
import pydantic  # noqa: E402

import weather as W  # noqa: E402
from agents import AgentError, AgentMessage, ModeratorAgent, PilotAgent, StakeholderVote, reply_schema  # noqa: E402
from agents.moderator import ConsensusSummary, ModeratorMessage  # noqa: E402
from baseline_agents import make_rule_agents  # noqa: E402
from constraints import Proposal, check_all, check_atc, check_ground, check_pilot, deterministic_plan, is_safe  # noqa: E402
from escalation import Alarm, dispatch  # noqa: E402
from monitoring import CallMetric  # noqa: E402
from negotiation import MAX_ROUNDS, MAX_TURNS, Autonomy, ConsensusLog, NegotiationSession, negotiate  # noqa: E402
from scenario import LIMITS, MODELLED_STANDS, WAKE_SEPARATION_MIN, build_scenario, load_disrupted_flights, split_of  # noqa: E402

DF = load_disrupted_flights(limit=60)


def scenario(i=40):
    return build_scenario(DF.iloc[i], weather_mode="OFF")


class Stub:
    """Replays scripted replies. Items: (stance, proposal|None, answering_plan|None) or an Exception."""
    model, timeout_s = "stub", 1.0

    def __init__(self, domain, script, votes=()):
        self.domain, self.script, self.calls = domain, list(script), 0
        self.votes, self.buses, self.briefings = list(votes) or [(True, "Agreed.")], [], []

    def ratify(self, s, turn, plan_id, briefing, own_report):
        self.briefings.append(briefing)
        item = self.votes.pop(0) if len(self.votes) > 1 else self.votes[0]
        if isinstance(item, Exception):
            raise item
        return StakeholderVote(agree=item[0], reason=item[1], agent=self.domain, turn=turn)

    def act(self, s, turn, bus, on_table, plan_id, own_report):
        self.buses.append(bus)
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        stance, proposal, ref = (item + (None, None))[:3]
        return AgentMessage(stance=stance, answering_plan=plan_id if ref is None else ref,
                            proposal=proposal or on_table, constraints=[], concession="-", rationale="scripted",
                            agent=self.domain, turn=turn)


def agents_with(atc, pilot, ground):
    return {"ATC": Stub("ATC", atc), "PILOT": Stub("PILOT", pilot), "GROUND": Stub("GROUND", ground)}


class TestData(unittest.TestCase):
    def test_filter_matches_challenge_rule(self):
        self.assertTrue(((DF.WEATHER_DELAY > 15) | (DF.LATE_AIRCRAFT_DELAY > 20)).all())

    def test_split_is_stable_and_roughly_20pct(self):
        splits = [split_of(r) for _, r in DF.iterrows()]
        self.assertEqual(splits, [split_of(r) for _, r in DF.iterrows()])
        self.assertTrue(0.05 < splits.count("test") / len(splits) < 0.4)

    def test_scenario_is_deterministic(self):
        self.assertEqual(scenario(5).model_dump(), scenario(5).model_dump())


class TestWeatherParsing(unittest.TestCase):
    def test_runway_wind_components(self):
        end = W.RunwayEnd(ident="27", heading_true=270, length_ft=10000, physical="09/27")
        head = W.runway_winds([end], 270, 20, None)[0]
        self.assertAlmostEqual(head.headwind_kt, 20, places=0)
        cross = W.runway_winds([end], 180, 30, 40)[0]
        self.assertAlmostEqual(cross.crosswind_kt, 40, places=0)
        self.assertFalse(cross.within_limits)
        tail = W.runway_winds([end], 90, 15, None)[0]
        self.assertGreater(tail.tailwind_kt, W.MAX_TAILWIND_KT)

    def test_flight_category_and_metar_fields(self):
        self.assertEqual(W._flight_category(0.5, 3000), "LIFR")
        self.assertEqual(W._flight_category(10, 800), "IFR")
        self.assertEqual(W._ceiling("KORD 011251Z 24011KT 10SM SCT020 BKN045 OVC090"), 4500)
        self.assertEqual(W._visibility("KORD 011251Z 24011KT 1 1/2SM BR"), 1.5)
        self.assertEqual(W._intensity("CONT LGT-MOD CHOP"), "MOD")


class TestValidators(unittest.TestCase):
    def test_reference_plan_is_safe(self):
        s = scenario()
        ref = deterministic_plan(s)
        self.assertEqual(ref.mode, "OPTIMISED_SOP")
        self.assertTrue(is_safe(check_all(s, ref.proposal)))
        self.assertTrue(all(is_safe(check_all(s, p)) for p, _ in ref.alternatives))

    def test_wake_separation_violation(self):
        s = scenario()
        rwy = s.runways[0]
        other = next(t for t in rwy.traffic if t.t >= s.eta_min)
        self.assertFalse(check_atc(s, Proposal(landing_slot_min=other.t, runway=rwy.id, gate="R1")).hard_ok)

    def test_fuel_reserve_violation(self):
        s = scenario()
        p = Proposal(landing_slot_min=s.latest_safe_landing + 1, runway=s.runways[0].id, gate="R1")
        self.assertFalse(check_pilot(s, p).hard_ok)

    def test_crosswind_out_of_limits_blocks_runway(self):
        s = scenario()
        s.runways[0].wind_within_limits = False
        p = deterministic_plan(s).proposal
        self.assertNotEqual(p.runway, s.runways[0].id)

    def test_starved_aircraft_diverts(self):
        s = scenario().model_copy(update={"fob_at_eta_kg": 100})
        self.assertEqual(deterministic_plan(s).mode, "DIVERT")

    def test_reply_schema_rejects_invented_ids(self):
        s = scenario()
        schema = reply_schema(s)
        good = {"stance": "ACCEPT", "answering_plan": 1, "constraints": [], "concession": "", "rationale": "",
                "proposal": {"landing_slot_min": 10, "runway": s.runways[0].id, "gate": "A1"}}
        schema.model_validate(good)
        bad = {**good, "proposal": {**good["proposal"], "runway": "99X"}}
        with self.assertRaises(pydantic.ValidationError):
            schema.model_validate(bad)


class TestNegotiation(unittest.TestCase):
    def setUp(self):
        self.s = scenario()
        self.safe = deterministic_plan(self.s).proposal
        patcher = mock.patch("negotiation.dispatch", side_effect=lambda a: a)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_with(self, atc, pilot, ground, **kw):
        agents = agents_with(atc, pilot, ground)
        return negotiate(self.s, agents, save_log=False, **kw), agents

    def test_consensus_on_safe_plan(self):
        out, _ = self.run_with([("PROPOSE", self.safe)], [("ACCEPT",)], [("ACCEPT",)])
        self.assertEqual((out.status, out.turns_used), ("CONSENSUS", 1))
        self.assertTrue(out.log_verified)

    def test_turn_limit_then_fallback(self):
        later = [Proposal(landing_slot_min=self.safe.landing_slot_min + i, runway=self.safe.runway, gate="R1")
                 for i in range(3)]
        out, agents = self.run_with([("PROPOSE", p) for p in later], [("COUNTER", self.safe)], [("ACCEPT",)])
        self.assertIn(out.status, ("FALLBACK", "HUMAN_DECISION"))
        self.assertLessEqual(agents["ATC"].calls, MAX_TURNS)

    def test_unsafe_accept_is_overridden(self):
        s = self.s
        late = next(t for t in range(s.latest_safe_landing + 2, s.latest_safe_landing + 90)
                    if check_atc(s, Proposal(landing_slot_min=t, runway=s.runways[0].id, gate="R1")).hard_ok)
        unsafe = Proposal(landing_slot_min=late, runway=s.runways[0].id, gate="R1")
        out, _ = self.run_with([("PROPOSE", unsafe)], [("ACCEPT",)], [("ACCEPT",)])
        self.assertNotEqual(out.status, "CONSENSUS")
        self.assertGreaterEqual(out.safety_overrides, 1)

    def test_stale_reply_is_discarded(self):
        out, _ = self.run_with([("PROPOSE", self.safe)], [("ACCEPT", None, 0)], [("ACCEPT",)])
        self.assertGreaterEqual(out.stale_replies, 1)
        self.assertNotEqual((out.status, out.turns_used), ("CONSENSUS", 1))

    def test_deadlock_detected_and_broken_early(self):
        other = Proposal(landing_slot_min=self.safe.landing_slot_min + 1, runway=self.safe.runway, gate="R1")
        out, agents = self.run_with([("PROPOSE", self.safe)], [("COUNTER", other)], [("ACCEPT",)])
        self.assertTrue(out.deadlock)
        self.assertIn("DEADLOCK", [a.code for a in out.alarms])
        self.assertEqual(agents["ATC"].calls, 2)  # stopped at turn 2 instead of burning turn 3

    def test_timeout_retry_then_success(self):
        out, agents = self.run_with([AgentError("ATC", "timed out", timeout=True), ("PROPOSE", self.safe)],
                                    [("ACCEPT",)], [("ACCEPT",)])
        self.assertEqual(out.status, "CONSENSUS")
        self.assertIn(("AGENT_TIMEOUT", "retry"), [(d.code, d.option) for d in out.decisions])
        self.assertEqual(agents["ATC"].timeout_s, 2.0)  # retry got double the time

    def test_timeout_keep_partial(self):
        out, _ = self.run_with([("PROPOSE", self.safe)], [("COUNTER", self.safe)],
                               [AgentError("GROUND", "timed out", timeout=True)],
                               autonomy=Autonomy.SUPERVISED,
                               decide=lambda req: "keep_partial" if req.code == "AGENT_TIMEOUT" else req.default)
        self.assertEqual(out.status, "HUMAN_DECISION")
        self.assertTrue(is_safe(check_all(self.s, out.proposal)))

    def test_step_by_step_pauses_and_human_choice_applies(self):
        later = Proposal(landing_slot_min=self.safe.landing_slot_min + 1, runway=self.safe.runway, gate="R1")
        session = NegotiationSession(self.s, agents_with([("PROPOSE", self.safe)], [("COUNTER", later)], [("ACCEPT",)]),
                                     Autonomy.STEP_BY_STEP, save_log=False)
        kind, data = session.advance()
        while kind != "decision":
            kind, data = session.advance()
        self.assertEqual(data.code, "TURN_REVIEW")
        while kind != "outcome":
            kind, data = session.advance("fallback" if kind == "decision" and data.code == "TURN_REVIEW"
                                         else (data.default if kind == "decision" else None))
        self.assertIn(data.status, ("FALLBACK", "HUMAN_DECISION"))
        self.assertTrue(any(d.by == "HUMAN" for d in data.decisions))

    def test_safety_drill_rogue_agents_never_get_an_unsafe_plan(self):
        from baseline_agents import make_drill_agents
        out = negotiate(self.s, make_drill_agents(), save_log=False)
        self.assertNotEqual(out.status, "CONSENSUS")
        self.assertGreaterEqual(out.safety_overrides, 1)
        self.assertTrue(out.proposal.runway == "DIVERT" or is_safe(check_all(self.s, out.proposal))
                        or out.reference.mode != "OPTIMISED_SOP")

    def test_rule_agents_reach_safe_outcome(self):
        out = negotiate(self.s, make_rule_agents(), save_log=False)
        self.assertTrue(out.status != "CONSENSUS" or is_safe(check_all(self.s, out.proposal)))


class ScriptedChief(ModeratorAgent):
    """Chief of Staff with canned text, recording what it was told."""

    def __init__(self, fail_summary=False):
        super().__init__(None, model="stub")
        self.fail_summary, self.objections = fail_summary, []

    def brief(self, s, turn, bus, opening, support="", remaining=0, objections=""):
        if objections:
            self.objections.append((objections, remaining))
        return ModeratorMessage(summary="brief", open_conflicts=[], bridge_proposal=None, directives=[], turn=turn)

    def summarise_consensus(self, s, turn, plan, positions, support=""):
        if self.fail_summary:
            raise AgentError("MODERATOR", "timed out", timeout=True)
        metric = CallMetric(agent="MODERATOR", turn=turn, model="stub", latency_s=0)
        return ConsensusSummary(summary=f"Agreed: {plan}", chief_reasoning="Cheapest safe plan."), metric


class TestRatification(unittest.TestCase):
    """A consensus goes to the stakeholders; a rejection earns one new round; a second rejection falls back."""

    def setUp(self):
        self.s = scenario()
        ref = deterministic_plan(self.s)
        self.first, self.second = ref.proposal, ref.alternatives[0][0]
        patcher = mock.patch("negotiation.dispatch", side_effect=lambda a: a)
        patcher.start()
        self.addCleanup(patcher.stop)

    def agents(self, pilot_votes=(), ground_votes=(), atc_votes=(), chief=None):
        team = {"ATC": Stub("ATC", [("PROPOSE", self.first), ("PROPOSE", self.second)], atc_votes),
                "PILOT": Stub("PILOT", [("ACCEPT",)], pilot_votes),
                "GROUND": Stub("GROUND", [("ACCEPT",)], ground_votes)}
        return {**team, "MODERATOR": chief} if chief else team

    def test_all_agree_commits_the_plan(self):
        out = negotiate(self.s, self.agents(), save_log=False)
        self.assertEqual((out.status, out.rounds, out.proposal), ("CONSENSUS", 1, self.first))
        self.assertIn("ratified by the stakeholders (AI agents)", out.plan_source)
        self.assertEqual(len(out.ratifications[0].votes), 3)
        self.assertTrue(out.log_verified)

    def test_a_disagreement_starts_a_new_round_with_the_objections(self):
        chief = ScriptedChief()
        team = self.agents(pilot_votes=[(False, "Connection to LAX at risk."), (True, "Fixed.")], chief=chief)
        out = negotiate(self.s, team, save_log=False)
        self.assertEqual((out.status, out.rounds, out.proposal), ("CONSENSUS", 2, self.second))
        self.assertEqual(chief.objections, [("PILOT: Connection to LAX at risk.", MAX_TURNS)])
        self.assertIn("OBJECTION to plan #1: Connection to LAX at risk.", team["ATC"].buses[1])
        self.assertIn(f"ROUND 2 of {MAX_ROUNDS}", team["ATC"].buses[1])
        self.assertEqual([r.accepted for r in out.ratifications], [False, True])

    def test_second_rejection_falls_back_to_the_planner(self):
        team = self.agents(ground_votes=[(False, "Stand too far."), (False, "Still too far.")])
        out = negotiate(self.s, team, save_log=False)
        self.assertEqual((out.status, out.rounds), ("FALLBACK", MAX_ROUNDS))
        self.assertIn("deterministic planner", out.plan_source)
        self.assertIn("Still too far.", out.reason)
        self.assertEqual(out.proposal, deterministic_plan(self.s).proposal)

    def test_no_consensus_in_the_new_round_falls_back(self):
        later = Proposal(landing_slot_min=self.second.landing_slot_min + 1, runway=self.second.runway, gate="R1")
        team = {"ATC": Stub("ATC", [("PROPOSE", self.first), ("PROPOSE", self.second), ("PROPOSE", later),
                                    ("PROPOSE", self.second), ("PROPOSE", later)]),
                "PILOT": Stub("PILOT", [("ACCEPT",), ("COUNTER", later)], [(False, "No.")]),
                "GROUND": Stub("GROUND", [("ACCEPT",)])}
        out = negotiate(self.s, team, save_log=False)
        self.assertEqual(out.status, "FALLBACK")
        self.assertEqual(out.rounds, 2)
        self.assertEqual(out.turns_used, 1 + MAX_TURNS)

    def test_new_round_gets_fresh_turns_up_to_six_in_total(self):
        def counter(i):
            return Proposal(landing_slot_min=self.first.landing_slot_min + 50 + i, runway=self.first.runway, gate="R1")
        plans = [self.first if i == 2 else counter(i) for i in range(3)] + [counter(i) for i in range(3, 6)]

        class AcceptsOnlyFirst(Stub):  # counters everything else, so each round runs to its last turn
            def act(inner, s, turn, bus, on_table, plan_id, own_report):
                inner.script = [("ACCEPT",) if on_table == self.first else ("COUNTER",)]
                return super().act(s, turn, bus, on_table, plan_id, own_report)

        team = {"ATC": Stub("ATC", [("PROPOSE", p) for p in plans]),
                "PILOT": AcceptsOnlyFirst("PILOT", [], [(False, "No.")]), "GROUND": Stub("GROUND", [("ACCEPT",)])}
        team["PILOT"].calls = 0
        out = negotiate(self.s, team, save_log=False)
        self.assertEqual(out.rounds, MAX_ROUNDS)
        self.assertEqual(out.turns_used, MAX_TURNS * MAX_ROUNDS)
        self.assertEqual(team["ATC"].calls, MAX_TURNS * MAX_ROUNDS)

    def test_rejected_plan_is_never_kept_as_the_best_so_far(self):
        team = {"ATC": Stub("ATC", [("PROPOSE", self.first)]),  # re-tables the rejected plan: a deadlock
                "PILOT": Stub("PILOT", [("ACCEPT",)], [(False, "No.")]), "GROUND": Stub("GROUND", [("ACCEPT",)])}
        session = NegotiationSession(self.s, team, save_log=False)
        kind, data = session.advance()
        while kind != "outcome":
            kind, data = session.advance()
        self.assertTrue(data.deadlock)
        self.assertNotIn("best safe plan", data.plan_source)

    def test_people_vote_when_supervised(self):
        chief = ScriptedChief()
        team = self.agents(chief=chief)
        session = NegotiationSession(self.s, team, Autonomy.SUPERVISED, save_log=False)
        kind, req = session.advance()
        while kind != "ratification":
            kind, req = session.advance()
        self.assertEqual(req.chief_summary, f"Agreed: {req.plan_text}")
        self.assertEqual(req.chief_reasoning, "Cheapest safe plan.")
        self.assertEqual({p.agent for p in req.positions}, {"ATC", "PILOT", "GROUND"})
        self.assertTrue(req.facts[0].endswith("hard checks pass."))
        kind, data = session.advance({d: {"agree": True, "reason": ""} for d in ("ATC", "PILOT", "GROUND")})
        while kind != "outcome":
            kind, data = session.advance()
        self.assertEqual(data.status, "CONSENSUS")
        self.assertIn("(people)", data.plan_source)
        self.assertEqual([d.by for d in data.decisions if d.code == "RATIFICATION"], ["HUMAN"] * 3)
        self.assertEqual(team["PILOT"].briefings, [])  # no AI call when the person answered
        actors = [e.actor for e in session.log.entries if e.kind == "stakeholder_vote"]
        self.assertEqual(actors, ["ATC_HUMAN", "PILOT_HUMAN", "GROUND_HUMAN"])

    def test_human_disagreement_reaches_the_new_round(self):
        team = self.agents()
        votes = {"ATC": {"agree": True}, "PILOT": {"agree": True}, "GROUND": {"agree": False, "reason": "Tug is broken."}}
        answers = [votes, None]
        out = negotiate(self.s, team, autonomy=Autonomy.SUPERVISED, save_log=False, ratify=lambda req: answers.pop(0))
        self.assertEqual(out.rounds, 2)
        self.assertIn("Tug is broken.", team["ATC"].buses[1])

    def test_a_person_left_out_is_covered_by_their_agent(self):
        team = self.agents(pilot_votes=[(True, "AI says fine.")])
        out = negotiate(self.s, team, autonomy=Autonomy.SUPERVISED, save_log=False,
                        ratify=lambda req: {"ATC": {"agree": True}, "GROUND": {"agree": True}})
        self.assertEqual(out.status, "CONSENSUS")
        self.assertIn("people and AI agents", out.plan_source)
        self.assertEqual({v.agent: v.by for v in out.ratifications[0].votes},
                         {"ATC": "HUMAN", "PILOT": "AGENT", "GROUND": "HUMAN"})

    def test_an_agent_that_cannot_answer_has_not_agreed(self):
        team = self.agents(pilot_votes=[AgentError("PILOT", "timed out", timeout=True), (True, "Back.")])
        out = negotiate(self.s, team, save_log=False)
        self.assertFalse(out.ratifications[0].accepted)
        self.assertIn("Did not answer", out.ratifications[0].votes[1].reason)
        self.assertEqual(out.rounds, 2)

    def test_chief_summary_failure_does_not_block_ratification(self):
        session = NegotiationSession(self.s, self.agents(chief=ScriptedChief(fail_summary=True)), save_log=False)
        kind, data = session.advance()
        while kind != "outcome":
            kind, data = session.advance()
        self.assertEqual(data.status, "CONSENSUS")
        self.assertIn("moderator_unavailable", [e.kind for e in session.log.entries])
        request = next(e for e in session.log.entries if e.kind == "ratification_request").payload
        self.assertIsNone(request["chief_summary"])
        self.assertTrue(request["facts"])

    def test_ai_stakeholders_are_not_shown_other_domains_private_figures(self):
        team = self.agents(chief=ScriptedChief())
        negotiate(self.s, team, save_log=False)
        briefing = team["GROUND"].briefings[0]
        self.assertNotIn("fuel margin", briefing)
        self.assertNotIn("Ground overtime", briefing)
        self.assertIn("Chief of Staff's reasoning: Cheapest safe plan.", briefing)
        self.assertIn("- PILOT (ACCEPT)", briefing)

    def test_votes_and_objections_are_in_the_hash_chained_log(self):
        team = self.agents(pilot_votes=[(False, "Connection at risk."), (True, "OK.")])
        session = NegotiationSession(self.s, team, save_log=False)
        kind, data = session.advance()
        while kind != "outcome":
            kind, data = session.advance()
        kinds = [e.kind for e in session.log.entries]
        self.assertEqual(kinds.count("ratification_request"), 2)
        self.assertEqual(kinds.count("stakeholder_vote"), 6)
        self.assertIn("stakeholder_objection", kinds)
        self.assertTrue(session.log.verify())

    def test_malformed_model_reply_becomes_an_agent_failure_not_a_crash(self):
        try:
            StakeholderVote.model_validate({})
        except pydantic.ValidationError as e:
            error = e
        client = mock.MagicMock()
        client.with_options.return_value.beta.messages.parse.side_effect = error
        with self.assertRaises(AgentError):
            PilotAgent(client).ratify(self.s, 1, 1, "brief", check_pilot(self.s, self.first))


class TestOperationalLimits(unittest.TestCase):
    """The organisers' workbook drives the scenario and the hard checks."""

    def by_airport(self, iata, wake=None):
        df = load_disrupted_flights(limit=None)
        rows = df[df.DESTINATION_AIRPORT == iata]
        if wake:
            rows = rows[(rows.DISTANCE >= 2000) == (wake == "HEAVY")]
        return build_scenario(rows.iloc[0], weather_mode="OFF")

    def test_workbook_values_are_loaded(self):
        ord_limits = LIMITS.airport("ORD")
        self.assertEqual((ord_limits.size_class, ord_limits.active_runways, ord_limits.gates,
                          ord_limits.max_runway_ops_per_hour, ord_limits.min_turnaround_min, ord_limits.spare_gates),
                         ("Large hub", 4, 150, 90, 45, 3))
        self.assertEqual((LIMITS.final_reserve_min, LIMITS.contingency_pct, LIMITS.spare_fuel_min), (30, 5, 45))
        self.assertEqual(LIMITS.hold_kg_per_hour, {"narrowbody": 2400, "widebody": 6500})
        self.assertEqual((LIMITS.flight_crew_max_duty_h, LIMITS.ground_shift_h, LIMITS.overtime_multiplier,
                          LIMITS.gate_change_notice_min), (13, 8, 1.5, 20))

    def test_wake_table_converts_nautical_miles_to_minutes(self):
        self.assertEqual(LIMITS.wake_nm[("SUPER", "SMALL")], 8)
        self.assertEqual(LIMITS.wake_nm[("LARGE", "HEAVY")], 3)  # "Any other pair"
        self.assertEqual(WAKE_SEPARATION_MIN[("SUPER", "SMALL")], 4)  # 8 NM at 140 kt, rounded up
        self.assertEqual(WAKE_SEPARATION_MIN[("HEAVY", "LARGE")], 3)
        self.assertEqual(WAKE_SEPARATION_MIN[("LARGE", "LARGE")], 2)

    def test_scenario_uses_the_airports_runways_stands_and_turnaround(self):
        s = self.by_airport("ORD")
        self.assertLessEqual(len(s.runways), 4)
        self.assertGreater(len(s.runways), 2)
        self.assertEqual(len([g for g in s.gates if not g.remote]), MODELLED_STANDS)
        free_at_eta = [g for g in s.gates if not g.remote and g.free_from <= s.eta_min]
        self.assertEqual(len(free_at_eta), 3)
        self.assertGreaterEqual(s.turnaround_min, 45)

    def test_fuel_follows_the_workbook(self):
        s = self.by_airport("ORD", "LARGE")
        self.assertEqual(s.hold_burn_kg_per_min, 40)
        self.assertEqual(s.final_reserve_kg, 30 * 40)
        self.assertGreater(s.contingency_fuel_kg, 0)
        self.assertEqual(s.min_landing_fuel_kg, s.final_reserve_kg + s.alternate_fuel_kg + s.contingency_fuel_kg)
        self.assertEqual(s.latest_safe_landing - s.eta_min, 45)  # the assumed spare fuel

    def test_gate_change_needs_notice(self):
        s = scenario()
        rwy = s.runways[0].id
        other = next(g.id for g in s.gates if g.id != s.scheduled_gate)
        check = lambda slot, gate: next(c for c in check_ground(s, Proposal(landing_slot_min=slot, runway=rwy, gate=gate)).checks  # noqa: E731
                                        if c.name == "gate_change_notice")
        self.assertFalse(check(s.decision_time_min + 19, other).passed)
        self.assertTrue(check(s.decision_time_min + 20, other).passed)
        self.assertTrue(check(s.decision_time_min + 1, s.scheduled_gate).passed)

    def test_flight_crew_duty_is_a_hard_limit_but_does_not_divert(self):
        s = scenario()
        ok = deterministic_plan(s).proposal
        self.assertTrue(check_pilot(s, ok).hard_ok)
        tired = s.model_copy(update={"flight_duty_end_min": s.eta_min - 60})
        self.assertIn("flight_crew_duty", [c.name for c in check_pilot(tired, ok).violations()])
        self.assertNotEqual(deterministic_plan(tired).mode, "DIVERT")

    def test_runway_capacity_is_a_hard_limit(self):
        s = scenario()
        ok = deterministic_plan(s).proposal
        low = s.airport_limits.model_copy(update={"max_runway_ops_per_hour": 1})
        full = s.model_copy(update={"airport_limits": low})
        self.assertIn("runway_capacity", [c.name for c in check_atc(full, ok).violations()])

    def test_overtime_is_a_cost_and_follows_the_shift_change(self):
        s = scenario()
        rwy = s.runways[0]
        slot = s.crew_shift_end_min - rwy.taxi_min["R"] - 1  # the turnaround starts just before the shift change
        report = check_ground(s, Proposal(landing_slot_min=slot, runway=rwy.id, gate="R1"))
        self.assertGreater(report.metrics["crew_overtime_min"], 0)
        self.assertNotIn("crew_overtime", [c.name for c in report.violations()])
        self.assertEqual(s.shift_end_at(s.crew_shift_end_min), s.crew_shift_end_min + 480)

    def test_every_consensus_passes_the_workbook_limits(self):
        with mock.patch("negotiation.dispatch", side_effect=lambda a: a):
            for i in range(0, 60, 6):
                s = scenario(i)
                out = negotiate(s, make_rule_agents(), save_log=False)
                self.assertLessEqual(out.turns_used, MAX_TURNS)
                if out.status == "CONSENSUS":
                    self.assertTrue(is_safe(check_all(s, out.proposal)))

    def test_log_records_the_limits_source(self):
        with mock.patch("negotiation.dispatch", side_effect=lambda a: a):
            session = NegotiationSession(scenario(), make_rule_agents(), save_log=False)
            session.advance()
        source = session.log.entries[0].payload["limits_source"]
        self.assertEqual((source["workbook"], source["sha256"]), (LIMITS.source, LIMITS.sha256))


class TestEmergencyAndCosts(unittest.TestCase):
    def setUp(self):
        from emergency import plan_emergency
        self.plan, self.s = plan_emergency, scenario()

    def test_emergency_is_fast_and_lands_at_earliest_feasible_slot(self):
        s = self.s
        for kind in ("FUEL", "MEDICAL", "TECHNICAL"):
            plan = self.plan(s, kind)
            self.assertLess(plan.decision_ms, 200)
            rwy = s.runway(plan.proposal.runway)
            self.assertTrue(rwy.wind_within_limits)
            earliest = min(max(r.closed_until, s.eta_min) for r in s.runways if r.wind_within_limits)
            if kind != "TECHNICAL":  # zero added holding: no slot later than the earliest feasible one
                self.assertEqual(plan.proposal.landing_slot_min, earliest)
            self.assertGreaterEqual(plan.proposal.landing_slot_min, max(rwy.closed_until, s.eta_min))
            self.assertTrue(all(plan.actions[d] for d in ("ATC", "PILOT", "GROUND")))

    def test_technical_goes_to_remote_stand_with_fire_and_rescue(self):
        plan = self.plan(self.s, "TECHNICAL")
        self.assertEqual(plan.proposal.gate, "R1")
        self.assertIn("ARFF", plan.actions["GROUND"][0])

    def test_emergency_diverts_when_no_runway_is_usable(self):
        s = self.s.model_copy(deep=True)
        for r in s.runways:
            r.wind_within_limits = False
        plan = self.plan(s, "FUEL")
        self.assertEqual(plan.divert_to, s.alternate)
        self.assertIsNone(plan.proposal)

    def test_costs_rank_only_safe_plans_and_are_itemised(self):
        from constraints import cost_breakdown, plan_robustness
        ref = deterministic_plan(self.s)
        items = cost_breakdown(self.s, ref.proposal)
        self.assertAlmostEqual(sum(items.values()), ref.cost, places=1)
        self.assertGreaterEqual(plan_robustness(self.s, ref.proposal, [p for p, _ in ref.alternatives]), 0.0)
        unsafe = Proposal(landing_slot_min=self.s.latest_safe_landing + 30, runway=self.s.runways[0].id, gate="R1")
        self.assertEqual(plan_robustness(self.s, unsafe, [ref.proposal]), 0.0)


class TestStressSuite(unittest.TestCase):
    def test_all_real_world_stress_cases_hold_the_invariants(self):
        import stress
        results = stress.run_suite(weather_mode="OFF")
        failed = {r.case: [i.name for i in r.invariants if not i.passed] for r in results if not r.passed}
        self.assertEqual(failed, {})
        self.assertEqual(len(results), 7)

    def test_time_budget_forces_a_decision(self):
        import stress
        s = scenario()
        slow = {k: stress.FlakyAgent(a, (0.6, 0.6)) for k, a in make_rule_agents().items()}
        out = negotiate(s, slow, save_log=False, deadline_s=1.0)
        self.assertTrue(out.deadline_hit)
        self.assertIn("TIME_BUDGET", [a.code for a in out.alarms])


class TestFailureAnalysis(unittest.TestCase):
    def test_fallback_runs_are_explained(self):
        import tempfile
        from pathlib import Path
        import failures as F
        from baseline_agents import make_drill_agents
        s = scenario()
        with tempfile.TemporaryDirectory() as d, mock.patch("negotiation.LOG_DIR", Path(d)), \
                mock.patch.object(F, "LOG_DIR", Path(d)), mock.patch("negotiation.dispatch", side_effect=lambda a: a):
            negotiate(s, make_drill_agents())
            runs = F.load_runs()
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0].failed)
        a = F.autopsy(runs[0])
        self.assertIn(a.primary_cause, ("Fuel reserve", "Unsafe ACCEPT overridden", "Deadlock (no movement)"))
        self.assertTrue(a.deadlock_point)


class TestEscalationAndLog(unittest.TestCase):
    def test_webhook_called(self):
        with mock.patch.dict("os.environ", {"ESCALATION_WEBHOOK_URL": "https://example.invalid/hook"}), \
                mock.patch("escalation.requests.post") as post:
            post.return_value.status_code = 200
            alarm = dispatch(Alarm(code="DEADLOCK", severity="ALARM", title="t", detail="d", flight="X1", turn=2))
        post.assert_called_once()
        self.assertIn("webhook:200", alarm.channels)

    def test_tamper_detection(self):
        log = ConsensusLog()
        log.add(1, "ATC", "message", {"slot": 30})
        log.add(1, "PILOT", "message", {"stance": "ACCEPT"})
        self.assertTrue(log.verify())
        log.entries[0].payload["slot"] = 99
        self.assertFalse(log.verify())


if __name__ == "__main__":
    unittest.main()
