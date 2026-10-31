"""Offline tests: data, weather parsing, validators, planner, negotiation engine, HITL gates, escalation, log.

Agents are scripted stubs and scenarios use weather_mode="OFF", so no API key, credits or network needed.
Run: .venv/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pydantic  # noqa: E402

import weather as W  # noqa: E402
from agents import AgentError, AgentMessage, reply_schema  # noqa: E402
from baseline_agents import make_rule_agents  # noqa: E402
from constraints import Proposal, check_all, check_atc, check_pilot, deterministic_plan, is_safe  # noqa: E402
from escalation import Alarm, dispatch  # noqa: E402
from negotiation import MAX_TURNS, Autonomy, ConsensusLog, NegotiationSession, negotiate  # noqa: E402
from scenario import build_scenario, load_disrupted_flights, split_of  # noqa: E402

DF = load_disrupted_flights(limit=60)


def scenario(i=40):
    return build_scenario(DF.iloc[i], weather_mode="OFF")


class Stub:
    """Replays scripted replies. Items: (stance, proposal|None, answering_plan|None) or an Exception."""
    model, timeout_s = "stub", 1.0

    def __init__(self, domain, script):
        self.domain, self.script, self.calls = domain, list(script), 0

    def act(self, s, turn, bus, on_table, plan_id, own_report):
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
        session = NegotiationSession(self.s, agents_with([("PROPOSE", self.safe)], [("ACCEPT",)], [("ACCEPT",)]),
                                     Autonomy.STEP_BY_STEP, save_log=False)
        kind, data = session.advance()
        while kind != "decision":
            kind, data = session.advance()
        self.assertEqual(data.code, "CONSENSUS_REVIEW")
        while kind != "outcome":
            kind, data = session.advance("fallback" if kind == "decision" and data.code == "CONSENSUS_REVIEW"
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
