"""Stratified, held-out batch evaluation.

Rows are split 80/20 dev/test by a stable hash of the flight identity (scenario.split_of). Tune prompts on
`dev`; report on `test` only, so the reported numbers are not fitted to rows you looked at. Sampling is
stratified by disruption bucket (EDA: 88% late-aircraft, 9% weather, 3% both), so weather cases are not
drowned out.

  python batch_eval.py --agents rules --n 10            # baseline, free
  python batch_eval.py --split test --n 4 --workers 3   # Claude agents (spends credits: ~$0.2-0.5 per flight)
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from agents import DEFAULT_MODEL, make_agents
from baseline_agents import make_rule_agents
from negotiation import Autonomy, negotiate
from scenario import build_scenario, load_disrupted_flights, split_of

OUT = Path(__file__).parent / "eval"
BUCKETS = ["late_aircraft", "weather", "both"]


def sample(split: str, n_per_bucket: int, seed: int) -> pd.DataFrame:
    df = load_disrupted_flights(limit=None)
    df = df[[split_of(r) == split for _, r in df.iterrows()]] if len(df) < 20_000 else \
        df[df.apply(split_of, axis=1) == split]
    parts = [g.sample(min(n_per_bucket, len(g)), random_state=seed) for _, g in df.groupby("BUCKET")]
    return pd.concat(parts)


def run_one(row: pd.Series, agents_kind: str, model: str, weather: str) -> dict:
    s = build_scenario(row, weather_mode=weather)
    agents = make_rule_agents() if agents_kind == "rules" else make_agents(model=model)
    out = negotiate(s, agents, autonomy=Autonomy.AUTONOMOUS)
    gap = (out.cost - out.reference.cost) if out.cost is not None and out.reference.cost is not None else None
    return {
        "flight": s.flight, "date": s.date, "route": f"{s.origin}-{s.destination}", "bucket": row.BUCKET,
        "weather_delay": s.weather_delay_min, "late_aircraft_delay": s.late_aircraft_delay_min,
        "dest_weather": s.weather_at_destination, "status": out.status, "plan_source": out.plan_source,
        "turns": out.turns_used, "rounds": out.rounds, "deadlock": out.deadlock, "safety_overrides": out.safety_overrides,
        "stale_replies": out.stale_replies, "alarms": len(out.alarms), "cost": out.cost,
        "optimum_cost": out.reference.cost, "cost_gap": gap, "plan_safe": bool(out.reports) and all(
            r.hard_ok for r in out.reports.values()), "usd": out.usage["cost_usd"],
        "tokens": out.usage["input_tokens"] + out.usage["output_tokens"], "elapsed_s": out.elapsed_s,
        "log": out.log_path,
    }


def summarise(df: pd.DataFrame) -> str:
    n = len(df)
    cons = df[df.status == "CONSENSUS"]
    lines = [
        f"- flights: {n} ({', '.join(f'{b} {int((df.bucket == b).sum())}' for b in BUCKETS)})",
        f"- consensus within 3 turns: **{len(cons)}/{n} ({100 * len(cons) / max(n, 1):.0f}%)**, mean turns "
        f"{cons.turns.mean():.2f}" if len(cons) else f"- consensus: 0/{n}",
        f"- deterministic fallback: {int((df.status == 'FALLBACK').sum())}, deadlocks: {int(df.deadlock.sum())}, "
        f"safety overrides: {int(df.safety_overrides.sum())}, stale replies: {int(df.stale_replies.sum())}",
        f"- final plans passing every hard check: {int(df.plan_safe.sum())}/{n} (any other is a fuel-priority or "
        "divert SOP outcome, where ATC re-sequences traffic by procedure)",
        f"- cost gap vs deterministic optimum (agreed plans): mean {df.cost_gap.mean():.1f}, median {df.cost_gap.median():.1f}",
        f"- spend: ${df.usd.sum():.2f} total, ${df.usd.mean():.3f}/flight; latency p50 {df.elapsed_s.median():.0f}s, "
        f"max {df.elapsed_s.max():.0f}s",
        "", "| bucket | n | consensus | mean turns | mean cost gap |", "|---|---:|---:|---:|---:|",
    ]
    for b in BUCKETS:
        g = df[df.bucket == b]
        if len(g):
            c = g[g.status == "CONSENSUS"]
            lines.append(f"| {b} | {len(g)} | {100 * len(c) / len(g):.0f}% | "
                         f"{c.turns.mean() if len(c) else float('nan'):.2f} | {g.cost_gap.mean():.1f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=4, help="flights per bucket")
    ap.add_argument("--agents", default="claude", choices=["claude", "rules"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--weather", default="HISTORICAL", choices=["HISTORICAL", "OFF"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = sample(args.split, args.n, args.seed)
    print(f"Running {len(rows)} {args.split} flights with {args.agents} agents ({args.workers} in parallel)...")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, r, args.agents, args.model, args.weather) for _, r in rows.iterrows()]
        for f in as_completed(futures):
            try:
                r = f.result()
                results.append(r)
                print(f"  {r['flight']:>7} {r['route']} {r['bucket']:<13} {r['status']:<14} turns={r['turns']} "
                      f"gap={r['cost_gap']} ${r['usd']:.3f}")
            except Exception as e:  # one broken scenario must not kill the batch
                print(f"  ERROR: {type(e).__name__}: {e}")
    df = pd.DataFrame(results)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    OUT.mkdir(exist_ok=True)
    base = OUT / f"{args.agents}_{args.split}_{stamp}"
    df.to_csv(base.with_suffix(".csv"), index=False)
    report = (f"# Batch evaluation: {args.agents} agents, {args.split} split\n\n{datetime.now():%Y-%m-%d %H:%M}, "
              f"model {args.model if args.agents == 'claude' else 'n/a'}, seed {args.seed}\n\n{summarise(df)}\n")
    base.with_suffix(".md").write_text(report)
    base.with_suffix(".json").write_text(json.dumps(vars(args)))
    print("\n" + report + f"\nSaved {base}.csv/.md")


if __name__ == "__main__":
    main()
