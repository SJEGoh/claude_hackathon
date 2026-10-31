"""Headless runner.

  python run_negotiation.py --row 3                       # Claude agents, autonomous
  python run_negotiation.py --row 3 --autonomy SUPERVISED # pauses and asks you in the terminal
  python run_negotiation.py --row 3 --agents rules        # rule-based agents, no API cost
  python run_negotiation.py --row 3 --weather LIVE        # today's aviationweather.gov weather
"""
import argparse

from agents import DEFAULT_MODEL, make_agents, render_brief, render_message
from baseline_agents import make_rule_agents
from negotiation import Autonomy, DecisionRequest, negotiate
from scenario import build_scenario, load_disrupted_flights


def ask_terminal(req: DecisionRequest) -> str:
    print(f"\n*** DECISION NEEDED [{req.severity}] {req.title}")
    for line in req.briefing:
        print(f"    {line}")
    for i, o in enumerate(req.options, 1):
        print(f"  {i}. {o.label}: {o.description}")
    raw = input(f"Choose 1-{len(req.options)} [default: {req.default}]: ").strip()
    return req.options[int(raw) - 1].id if raw.isdigit() and 1 <= int(raw) <= len(req.options) else req.default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=0, help="index into the filtered disrupted-flight list")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--autonomy", default="AUTONOMOUS", choices=[a.value for a in Autonomy])
    ap.add_argument("--agents", default="claude", choices=["claude", "rules"])
    ap.add_argument("--weather", default="HISTORICAL", choices=["HISTORICAL", "LIVE", "OFF"])
    ap.add_argument("--no-moderator", action="store_true")
    args = ap.parse_args()

    s = build_scenario(load_disrupted_flights(limit=None).iloc[args.row], weather_mode=args.weather)
    print(f"{s.flight} {s.origin}->{s.destination} {s.date} SIBT {s.clock(0)}  WEATHER_DELAY={s.weather_delay_min} "
          f"LATE_AIRCRAFT_DELAY={s.late_aircraft_delay_min}\n{s.weather_attribution}\n")

    def on_event(kind, data):
        if kind == "turn":
            print(f"\n===== TURN {data['turn']} =====")
        elif kind == "message":
            print(render_message(data, s))
        elif kind == "brief":
            print(render_brief(data))
        elif kind == "system":
            print(f"  [SYSTEM {data['kind']}] {data['text']}")
        elif kind == "alarm":
            print(f"  !!! {data.severity} {data.code}: {data.title} ({data.detail}) -> {', '.join(data.channels)}")
        elif kind == "metric":
            print(f"  ({data.agent} {data.latency_s}s, {data.input_tokens}+{data.output_tokens} tok, ${data.cost_usd:.4f})")
        elif kind == "decision_auto":
            print(f"  [auto-decision] {data['request'].code} -> {data['choice']}")

    agents = (make_rule_agents() if args.agents == "rules" else
              make_agents(model=args.model, effort=args.effort, moderator=not args.no_moderator))
    out = negotiate(s, agents, on_event, autonomy=Autonomy(args.autonomy), decide=ask_terminal)
    p = out.proposal
    print(f"\nRESULT: {out.status} ({out.plan_source}) after {out.turns_used} turn(s): TLDT {s.clock(p.landing_slot_min)} "
          f"(slot {p.landing_slot_min}) RWY {p.runway}, stand {p.gate}")
    print(f"cost {out.cost} vs deterministic optimum {out.reference.cost} | {out.elapsed_s}s | "
          f"${out.usage['cost_usd']} ({out.usage['input_tokens']}+{out.usage['output_tokens']} tok) | "
          f"log {out.log_path} (chain {'OK' if out.log_verified else 'BROKEN'})")


if __name__ == "__main__":
    main()
