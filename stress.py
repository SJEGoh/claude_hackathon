"""Real-world stress tests for the negotiation framework.

Each case reproduces a documented ATM failure mode and checks the same safety invariants:
  no crash · never an unsafe agreed plan · at most 3 turns · inside the time budget · log hash chain intact

  NATS_RECORD        UK NATS, Aug 2023: one flight plan with two different beacons sharing a name crashed the
                     primary AND the identical backup (common-mode). Lesson: validate inputs, quarantine the bad
                     record, keep processing everyone else, and keep the fallback independent of the primary.
  LIGHTNING_RAMP     Lightning halts ramp work (HKIA: all ramp activity stops within 1 km; Changi suspends
                     baggage handling under lightning warnings, e.g. 4 May 2024). Aircraft land but cannot get
                     onto contact stands.
  RADAR_OUTAGE       Philadelphia TRACON / Newark, Apr 28 and May 9 2025: ~90 s loss of radar and radio over a
                     single data feed. Here: every AI agent is unreachable at once.
  STAFFING_SHORTAGE  US ATC ~3,500 controllers short (2025), six-day weeks: single-runway operation with extra
                     spacing, slow and flaky responders, and a hard time budget.
  RUNWAY_CLOSURE     Disabled aircraft closes the preferred runway for 45 minutes at the worst moment.
  ROGUE_AGENTS       Adversarial agents push a fuel-breaching plan (safety drill).
  STORM_FRONT_WAVE   Several real DOT flights into the same airport at once, negotiated in parallel on a stale
                     snapshot, committed through a lock-protected ResourceLedger with re-validation.
"""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import pandas as pd
from pydantic import BaseModel

from agents.base import AgentError
from baseline_agents import make_drill_agents, make_rule_agents
from constraints import Proposal, check_all, cost, deterministic_plan, is_safe
from negotiation import MAX_TURNS, Outcome, negotiate
from scenario import WAKE_SEPARATION_MIN, Scenario, TrafficSlot, build_scenario, load_disrupted_flights

SOURCES = {
    "NATS_RECORD": "https://www.caa.co.uk/our-work/publications/documents/content/cap2993/",
    "LIGHTNING_RAMP": "https://www.hongkongairport.com/en/media-centre/press-release/2025/pr_1807",
    "RADAR_OUTAGE": "https://www.accuweather.com/en/travel/air-traffic-controllers-lose-radar-and-radio-contact-with-planes-approaching-and-departing-newark-for-90-seconds-again/1773424",
    "STAFFING_SHORTAGE": "https://www.axios.com/2025/11/02/flight-delays-staffing-shortages-government-shutdown",
}


# ============================================================================
# 1. Input validation and quarantine (the NATS lesson)
# ============================================================================

REQUIRED = ["AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "YEAR", "MONTH",
            "DAY", "SCHEDULED_ARRIVAL", "DISTANCE", "WEATHER_DELAY", "LATE_AIRCRAFT_DELAY"]


def validate_row(row: pd.Series) -> list[str]:
    """Reasons a DOT record must not reach the agents. Empty list = clean."""
    issues = []
    for col in REQUIRED:
        if col not in row or pd.isna(row[col]):
            issues.append(f"missing {col}")
    if issues:
        return issues
    try:
        hhmm = int(row.SCHEDULED_ARRIVAL)
        if not (0 <= hhmm <= 2400 and hhmm % 100 < 60):
            issues.append(f"SCHEDULED_ARRIVAL {row.SCHEDULED_ARRIVAL} is not a valid HHMM time")
        for col in ("WEATHER_DELAY", "LATE_AIRCRAFT_DELAY", "DISTANCE"):
            if float(row[col]) < 0:
                issues.append(f"{col} is negative ({row[col]})")
        if not (1 <= int(row.MONTH) <= 12 and 1 <= int(row.DAY) <= 31):
            issues.append("invalid date")
    except (TypeError, ValueError) as e:
        issues.append(f"non-numeric field: {e}")
    if str(row.ORIGIN_AIRPORT) == str(row.DESTINATION_AIRPORT):
        issues.append("origin equals destination")
    return issues


def validate_scenario(s: Scenario) -> list[str]:
    """The exact NATS failure class: two different objects sharing one identifier."""
    issues = []
    for kind, ids in (("runway", [r.id for r in s.runways]), ("stand", [g.id for g in s.gates])):
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            issues.append(f"duplicate {kind} identifier(s) {sorted(dupes)}: ambiguous, cannot plan safely")
    if s.fob_at_eta_kg <= 0:
        issues.append("non-positive fuel on board")
    return issues


class Ingest(BaseModel):
    accepted: list[str]
    quarantined: dict[str, list[str]]


def ingest(rows: pd.DataFrame, weather_mode: str = "OFF") -> tuple[list[Scenario], Ingest]:
    """Build scenarios for a batch; bad records are quarantined, never allowed to stop the batch."""
    good, accepted, quarantined = [], [], {}
    for i, row in rows.iterrows():
        key = f"row {i} ({row.get('AIRLINE', '?')}{row.get('FLIGHT_NUMBER', '?')})"
        issues = validate_row(row)
        if not issues:
            try:
                s = build_scenario(row, weather_mode=weather_mode)
                issues = validate_scenario(s)
            except Exception as e:  # noqa: BLE001 - any build failure quarantines this record only
                issues = [f"build failed: {type(e).__name__}: {e}"]
        if issues:
            quarantined[key] = issues
        else:
            good.append(s)
            accepted.append(key)
    return good, Ingest(accepted=accepted, quarantined=quarantined)


# ============================================================================
# 2. Fault-injection wrappers and scenario injectors
# ============================================================================

class FlakyAgent:
    """Wraps an agent: adds latency and, with probability fail_rate, a timeout (fatigue / overload)."""

    def __init__(self, inner, delay_s: tuple[float, float] = (0.0, 0.0), fail_rate: float = 0.0, seed: int = 0):
        self.inner, self.delay_s, self.fail_rate = inner, delay_s, fail_rate
        self.rng = random.Random(seed)
        self.model, self.timeout_s, self.domain = getattr(inner, "model", "?"), 1.0, getattr(inner, "domain", "?")

    def act(self, *args, **kwargs):
        time.sleep(self.rng.uniform(*self.delay_s))
        if self.rng.random() < self.fail_rate:
            raise AgentError(self.domain, "timed out (injected)", timeout=True)
        return self.inner.act(*args, **kwargs)

    def ratify(self, *args, **kwargs):
        time.sleep(self.rng.uniform(*self.delay_s))
        if self.rng.random() < self.fail_rate:
            raise AgentError(self.domain, "timed out (injected)", timeout=True)
        return self.inner.ratify(*args, **kwargs)


class OutageAgent:
    """Every call fails: the agent's link is down (radar/comms outage analogue)."""
    model, timeout_s = "outage", 1.0

    def __init__(self, domain: str):
        self.domain = domain

    def act(self, *args, **kwargs):
        raise AgentError(self.domain, "network error: link down (injected outage)")

    ratify = act


def inject_lightning(s: Scenario, minutes: int = 40) -> Scenario:
    """Ramp closed from ETA for `minutes`: contact stands unusable until it reopens (remote bay still usable)."""
    s = s.model_copy(deep=True)
    reopen = s.eta_min + minutes
    s.ramp_closed_until = max(s.ramp_closed_until, reopen)
    for g in s.gates:
        if not g.remote:
            g.free_from = max(g.free_from, reopen)
    return s


def inject_runway_closure(s: Scenario, minutes: int = 45) -> Scenario:
    s = s.model_copy(deep=True)
    best = deterministic_plan(s).proposal.runway
    for r in s.runways:
        if r.id == best:
            r.closed_until = max(r.closed_until, s.eta_min + minutes)
            r.closure_reason = "disabled aircraft on the runway"
    return s


def inject_staffing(s: Scenario) -> Scenario:
    """Single-runway operation and IMC-style spacing (fewer controllers, reduced arrival rate)."""
    s = s.model_copy(deep=True)
    if len(s.runways) > 1:
        s.runways[1].closed_until = 10_000
        s.runways[1].closure_reason = "position closed: controller staffing"
    s.imc = True
    return s


# ============================================================================
# 3. Multi-flight wave with a lock-protected ResourceLedger
# ============================================================================

def _shift(t: int, delta: int) -> int:
    return t if t <= -999 or t >= 10_000 else t + delta


def _sibt(s: Scenario) -> int:
    return int(s.scheduled_arrival_hhmm[:2]) % 24 * 60 + int(s.scheduled_arrival_hhmm[2:])


class Commit(BaseModel):
    flight: str
    priority: bool = False  # fuel priority: background traffic was re-sequenced by ATC
    runway: str
    landing_abs: int
    gate: str
    stand_from_abs: int
    stand_to_abs: int
    wake: str


class ResourceLedger:
    """Shared airport resources for a wave. Agents never write it: only `commit` does, under a lock, after
    re-validating the plan against the LATEST state (optimistic concurrency, version-checked)."""

    def __init__(self, base: Scenario):
        self.base, self.lock, self.version, self.commits = base, threading.Lock(), 0, []
        self.rejections: list[str] = []

    def view(self, s: Scenario, background: bool = True) -> Scenario:
        """`s` re-expressed on the shared airport state (base runways and stands + everything committed).
        background=False drops the non-wave traffic (what ATC may re-sequence for a fuel-priority arrival)."""
        delta = _sibt(self.base) - _sibt(s)
        runways = []
        for r in self.base.runways:
            r2 = r.model_copy(deep=True)
            r2.closed_until = _shift(r.closed_until, delta)
            r2.traffic = [t.model_copy(update={"t": t.t + delta}) for t in r.traffic] if background else []
            r2.traffic += [TrafficSlot(t=c.landing_abs - _sibt(s), callsign=c.flight, wake=c.wake, source="DOT")
                           for c in self.commits if c.runway == r.id and c.flight != s.flight]
            runways.append(r2)
        gates = []
        for g in self.base.gates:
            g2 = g.model_copy(update={"free_from": _shift(g.free_from, delta), "free_until": _shift(g.free_until, delta)})
            for c in self.commits:  # conservative: a committed stand is usable again only after that turn ends
                if c.gate == g.id and c.flight != s.flight and not g.remote:
                    g2.free_from = max(g2.free_from, c.stand_to_abs - _sibt(s))
            gates.append(g2)
        return s.model_copy(update={"runways": runways, "gates": gates, "scheduled_gate": self.base.scheduled_gate})

    def commit(self, s: Scenario, p: Proposal, seen_version: int, priority: bool = False) -> tuple[bool, str]:
        with self.lock:
            v = self.view(s, background=not priority)
            reports = check_all(v, p)
            if not is_safe(reports):
                why = "; ".join(c.detail for r in reports.values() for c in r.violations())
                self.rejections.append(f"{s.flight}: plan made on v{seen_version}, ledger at v{self.version}: {why}")
                return False, why
            ob = reports["GROUND"].metrics["eibt_min"]
            end = reports["GROUND"].metrics["turnaround_end_min"]
            self.commits.append(Commit(flight=s.flight, priority=priority, runway=p.runway, landing_abs=_sibt(s) + p.landing_slot_min,
                                       gate=p.gate, stand_from_abs=_sibt(s) + ob, stand_to_abs=_sibt(s) + end,
                                       wake=s.wake))
            self.version += 1
            return True, "committed"


def find_wave(k: int = 5, window_min: int = 90) -> pd.DataFrame:
    """The first airport/day in the DOT data with >= k disrupted arrivals inside `window_min`."""
    df = load_disrupted_flights(limit=None)
    df = df[df.DESTINATION_AIRPORT.isin(["ORD", "ATL", "DFW", "DEN", "LAX", "SFO", "IAH"])]
    df = df.assign(SA=df.SCHEDULED_ARRIVAL // 100 * 60 + df.SCHEDULED_ARRIVAL % 100)
    for _, g in df.groupby(["DESTINATION_AIRPORT", "MONTH", "DAY"]):
        g = g.sort_values("SA")
        for i in range(len(g) - k + 1):
            if g.SA.iloc[i + k - 1] - g.SA.iloc[i] <= window_min:
                return g.iloc[i:i + k]
    raise ValueError("no wave found")


class WaveResult(BaseModel):
    airport: str
    commits: list[Commit]
    replans: int
    rejections: list[str]
    separation_ok: bool
    stands_ok: bool
    order: list[str]


def run_wave(rows: pd.DataFrame, make_agents: Callable[[], dict], weather_mode: str = "OFF") -> WaveResult:
    scenarios, report = ingest(rows, weather_mode)
    base = scenarios[0]
    ledger = ResourceLedger(base)
    # Real ATC priority: least fuel margin first, then earliest ETA.
    scenarios.sort(key=lambda s: ((s.fob_at_eta_kg - s.min_landing_fuel_kg) / s.hold_burn_kg_per_min,
                                  _sibt(s) + s.eta_min))
    snapshot = {s.flight: ledger.view(s) for s in scenarios}  # every negotiation starts from the SAME (stale) v0

    def negotiate_one(s: Scenario) -> tuple[Scenario, Outcome]:
        return s, negotiate(snapshot[s.flight], make_agents(), save_log=False)

    with ThreadPoolExecutor(max_workers=len(scenarios)) as pool:
        results = list(pool.map(negotiate_one, scenarios))
    replans = 0
    for s, out in results:  # commit in priority order; stale plans are re-validated and re-planned
        ok, _ = ledger.commit(s, out.proposal, seen_version=0)
        if not ok:
            replans += 1
            fresh = deterministic_plan(ledger.view(s))
            if fresh.mode == "OPTIMISED_SOP":
                ledger.commit(s, fresh.proposal, seen_version=ledger.version)
            else:  # saturated: fuel priority, re-sequencing background traffic but never another wave flight
                prio = deterministic_plan(ledger.view(s, background=False))
                ledger.commit(s, prio.proposal, seen_version=ledger.version, priority=True)
    sep_ok = True
    by_rwy: dict[str, list[Commit]] = {}
    for c in ledger.commits:
        by_rwy.setdefault(c.runway, []).append(c)
    for cs in by_rwy.values():
        cs.sort(key=lambda c: c.landing_abs)
        for a, b in zip(cs, cs[1:]):
            sep_ok &= b.landing_abs - a.landing_abs >= WAKE_SEPARATION_MIN[(a.wake, b.wake)]
    stands_ok = all(not (a.gate == b.gate and a.gate != "R1" and a.stand_from_abs < b.stand_to_abs
                         and b.stand_from_abs < a.stand_to_abs)
                    for i, a in enumerate(ledger.commits) for b in ledger.commits[i + 1:])
    return WaveResult(airport=base.destination, commits=ledger.commits, replans=replans,
                      rejections=ledger.rejections, separation_ok=sep_ok, stands_ok=stands_ok,
                      order=[s.flight for s in scenarios])


# ============================================================================
# 4. The suite
# ============================================================================

class Invariant(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class StressResult(BaseModel):
    case: str
    title: str
    basis: str
    source: Optional[str]
    status: str
    elapsed_s: float
    invariants: list[Invariant]
    notes: list[str]

    @property
    def passed(self) -> bool:
        return all(i.passed for i in self.invariants)


def _core_invariants(s: Scenario, out: Outcome, budget_s: Optional[float]) -> list[Invariant]:
    reports = check_all(s, out.proposal) if s.runway(out.proposal.runway) else {}
    sop = out.status != "CONSENSUS" and out.reference.mode != "OPTIMISED_SOP"
    safe = bool(reports) and is_safe(reports)
    return [
        Invariant(name="never an unsafe agreed plan", passed=safe or sop,
                  detail="passes every hard check" if safe else f"SOP outcome ({out.reference.mode})"),
        Invariant(name=f"at most {MAX_TURNS} turns per round", passed=out.turns_used <= MAX_TURNS * out.rounds,
                  detail=f"{out.turns_used} turns in {out.rounds} round(s)"),
        Invariant(name="inside time budget", passed=budget_s is None or out.elapsed_s <= budget_s + 5,
                  detail=f"{out.elapsed_s}s" + (f" / {budget_s:.0f}s" if budget_s else "")),
        Invariant(name="log hash chain intact", passed=out.log_verified),
    ]


def run_suite(engine: Callable[[], dict] = make_rule_agents, row: int = 40, weather_mode: str = "OFF",
              cases: Optional[list[str]] = None) -> list[StressResult]:
    df = load_disrupted_flights(limit=200)
    base = build_scenario(df.iloc[row], weather_mode=weather_mode)
    results: list[StressResult] = []

    def single(case, title, basis, s, agents, budget=None, extra: Callable[[Outcome], list[Invariant]] = lambda o: []):
        if cases and case not in cases:
            return
        t0 = time.monotonic()
        try:
            out = negotiate(s, agents, save_log=False, deadline_s=budget)
            inv = [Invariant(name="no crash", passed=True)] + _core_invariants(s, out, budget) + extra(out)
            status = f"{out.status} ({out.plan_source}), {len(out.alarms)} alarms"
            notes = [f"final: {out.proposal.model_dump()}", *(f"alarm {a.code}: {a.detail}" for a in out.alarms)]
        except Exception as e:  # noqa: BLE001 - a crash is a failed invariant, not a crashed suite
            inv, status, notes = [Invariant(name="no crash", passed=False, detail=f"{type(e).__name__}: {e}")], "CRASH", []
        results.append(StressResult(case=case, title=title, basis=basis, source=SOURCES.get(case), status=status,
                                    elapsed_s=round(time.monotonic() - t0, 2), invariants=inv, notes=notes))

    # NATS: corrupt records in a batch, one scenario with a duplicate identifier
    if not cases or "NATS_RECORD" in cases:
        t0 = time.monotonic()
        batch = df.iloc[[row, row + 1, row + 2, row + 3]].copy()
        batch.loc[batch.index[1], "SCHEDULED_ARRIVAL"] = 2575  # impossible HHMM
        batch.loc[batch.index[2], "WEATHER_DELAY"] = -30        # negative delay
        good, rep = ingest(batch, weather_mode)
        dup = base.model_copy(deep=True)
        dup.gates[3].id = dup.gates[4].id  # two different stands, one name: the NATS failure class
        dup_issues = validate_scenario(dup)
        results.append(StressResult(
            case="NATS_RECORD", title="Malformed / ambiguous record in a batch",
            basis="UK NATS Aug 2023: two different beacons with the same name crashed primary and backup.",
            source=SOURCES["NATS_RECORD"], status=f"{len(rep.accepted)} accepted, {len(rep.quarantined)} quarantined",
            elapsed_s=round(time.monotonic() - t0, 2),
            invariants=[Invariant(name="no crash", passed=True),
                        Invariant(name="bad records quarantined, not propagated", passed=len(rep.quarantined) == 2,
                                  detail="; ".join(f"{k}: {v[0]}" for k, v in rep.quarantined.items())),
                        Invariant(name="clean records still processed", passed=len(good) == 2),
                        Invariant(name="duplicate identifiers detected", passed=bool(dup_issues),
                                  detail=dup_issues[0] if dup_issues else "")],
            notes=["Fallback planner is independent code, not a copy of the agents (avoids common-mode failure)."]))

    single("LIGHTNING_RAMP", "Lightning: ramp closed 40 min from ETA",
           "HKIA stops all ramp work within 1 km; Changi suspends baggage handling under lightning warnings.",
           inject_lightning(base, 40), engine(),
           extra=lambda o: [Invariant(name="no contact stand before the ramp reopens",
                                      passed=o.proposal.gate == "R1" or (o.reports and
                                      o.reports["GROUND"].metrics.get("eibt_min", 10**6) >= inject_lightning(base, 40).ramp_closed_until)
                                      or o.status != "CONSENSUS")])
    single("RADAR_OUTAGE", "All AI agents unreachable", "PHL TRACON / Newark 2025: ~90 s loss of radar and radio.",
           base, {k: OutageAgent(k) for k in ("ATC", "PILOT", "GROUND")}, budget=30,
           extra=lambda o: [Invariant(name="independent fallback took over", passed=o.status != "CONSENSUS"),
                            Invariant(name="alarm raised", passed=any(a.severity == "ALARM" for a in o.alarms))])
    single("STAFFING_SHORTAGE", "Single runway, extra spacing, slow and flaky responders, 20 s budget",
           "US ATC ~3,500 controllers short in 2025; six-day weeks; ground stops.",
           inject_staffing(base), {k: FlakyAgent(a, (0.2, 1.5), 0.25, seed=7) for k, a in engine().items()}, budget=20)
    single("RUNWAY_CLOSURE", "Preferred runway closed 45 min by a disabled aircraft",
           "Runway closures after incidents (e.g. Changi's runway recovery procedures).",
           inject_runway_closure(base, 45), engine(),
           extra=lambda o: [Invariant(name="closed runway not used before reopening",
                                      passed=o.proposal.runway == "DIVERT" or not o.reports or o.reports["ATC"].hard_ok
                                      or o.status != "CONSENSUS")])
    single("ROGUE_AGENTS", "Adversarial agents push a fuel-breaching plan", "Safety drill.",
           base, make_drill_agents(),
           extra=lambda o: [Invariant(name="unsafe ACCEPT overridden", passed=o.safety_overrides >= 1)])

    if not cases or "STORM_FRONT_WAVE" in cases:
        t0 = time.monotonic()
        try:
            wave = run_wave(find_wave(5), engine, weather_mode)
            inv = [Invariant(name="no crash", passed=True),
                   Invariant(name="wake separation between committed flights", passed=wave.separation_ok),
                   Invariant(name="no stand double-booked", passed=wave.stands_ok),
                   Invariant(name="every flight committed", passed=len(wave.commits) == len(wave.order))]
            status = f"{len(wave.commits)} flights committed at {wave.airport}, {wave.replans} re-planned after stale-state rejection"
            prio = [c.flight for c in wave.commits if c.priority]
            notes = [f"priority order (least fuel margin first): {', '.join(wave.order)}",
                     f"fuel-priority landings (background traffic re-sequenced by ATC): {', '.join(prio) or 'none'}",
                     *wave.rejections]
        except Exception as e:  # noqa: BLE001
            inv, status, notes = [Invariant(name="no crash", passed=False, detail=f"{type(e).__name__}: {e}")], "CRASH", []
        results.append(StressResult(
            case="STORM_FRONT_WAVE", title="5 disrupted arrivals at one hub, negotiated in parallel",
            basis="Convective fronts compress arrivals into the remaining capacity; parallel decisions race.",
            source=None, status=status, elapsed_s=round(time.monotonic() - t0, 2), invariants=inv, notes=notes))
    return results


if __name__ == "__main__":
    import sys
    from datetime import datetime
    from pathlib import Path

    res = run_suite(weather_mode=sys.argv[1] if len(sys.argv) > 1 else "OFF")
    lines = ["# Stress test report", "", f"{datetime.now():%Y-%m-%d %H:%M} · rule-based agents", "",
             "| case | result | invariants | time |", "|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r.case} | {r.status} | {'✅ all pass' if r.passed else '❌ ' + ', '.join(i.name for i in r.invariants if not i.passed)} | {r.elapsed_s}s |")
    text = "\n".join(lines)
    print(text)
    out = Path(__file__).parent / "eval" / f"stress_{datetime.now():%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")
