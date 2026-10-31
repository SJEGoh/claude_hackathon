"""Failure analysis: negotiations that did not end in an agreed, ratified plan, and why."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import failures as F
from constraints import Proposal, check_all, cost, is_safe
from negotiation import Autonomy, negotiate
from scenario import build_scenario, load_disrupted_flights
from ui_common import ENGINES, hero, make_engine_agents

hero(compact=True)
st.subheader("🔍 Failure analysis")
st.caption("Every negotiation that ended in the deterministic planner or a human override instead of an agreed, "
           "ratified plan, read straight from the hash-chained consensus logs.")


@st.cache_resource(show_spinner="Rebuilding the scenario from the DOT record…", max_entries=32)
def scenario_for(dot_row: int, weather_mode: str):
    df = load_disrupted_flights(limit=None)
    match = df[df["index"] == dot_row]
    return build_scenario(match.iloc[0], weather_mode=weather_mode) if len(match) else None


def plan_eval(s, plan: dict) -> dict:
    """USD and holding fuel for a logged plan, re-checked against the rebuilt scenario."""
    if not plan or plan.get("runway") == "DIVERT" or s.runway(plan.get("runway", "")) is None:
        return {"safe": False, "usd": None, "fuel_kg": None, "label": f"divert to {plan.get('gate')}" if plan else "n/a"}
    p = Proposal(**plan)
    r = check_all(s, p)
    return {"safe": is_safe(r), "usd": cost(s, p, r) if is_safe(r) else None,
            "fuel_kg": r["PILOT"].metrics.get("holding_fuel_kg"),
            "label": f"{s.clock(p.landing_slot_min)} · RWY {p.runway} · stand {p.gate}"}


runs = F.load_runs()
failed = [r for r in runs if r.failed]
if not runs:
    st.info("No negotiation logs yet. Run a negotiation in CHATS or CONTROLS.")
    st.stop()

# ---------------------------------------------------------------- KPIs
autopsies = {r.file: F.autopsy(r) for r in failed}
excess_co2, excess_usd = 0.0, 0.0
for r in failed:
    s = scenario_for(r.dot_row, r.weather_mode) if r.dot_row is not None else None
    a = autopsies[r.file]
    if s and a.best_ai_plan and a.best_ai_safe:
        fin, best = plan_eval(s, r.final), plan_eval(s, a.best_ai_plan)
        if fin["fuel_kg"] is not None and best["fuel_kg"] is not None:
            excess_co2 += (fin["fuel_kg"] - best["fuel_kg"]) * F.CO2_PER_KG_FUEL
        if fin["usd"] is not None and best["usd"] is not None:
            excess_usd += fin["usd"] - best["usd"]
all_causes = pd.Series([a.primary_cause for a in autopsies.values()]).value_counts()
k = st.columns(4)
k[0].metric("Negotiations logged", len(runs))
k[1].metric("Fallback / override rate", f"{len(failed) / len(runs):.0%}", f"{len(failed)} of {len(runs)}",
            delta_color="off")
k[2].metric("Fallback vs best safe AI plan", f"{excess_co2:+,.0f} kg CO₂", f"{excess_usd:+,.0f} USD",
            delta_color="inverse", help="Positive: the fallback burned/cost more than the best safe plan the agents "
            "produced. Negative or zero: the deterministic planner (cheapest safe plan) did as well or better, so the "
            "cost of the failure was lost agreement and time, not fuel.")
k[3].metric("Primary cause", all_causes.index[0] if len(all_causes) else "–",
            f"{all_causes.iloc[0]} run(s)" if len(all_causes) else None, delta_color="off")

if not failed:
    st.success("Every logged negotiation ended in an agreed, ratified plan.")
    st.stop()

# ---------------------------------------------------------------- incident list
st.markdown("#### Incidents")
table = pd.DataFrame([{"when (UTC)": r.started[:16].replace("T", " "), "flight": r.flight, "route": r.route,
                       "date": r.date, "outcome": r.status.replace("_", " ").lower(), "turns": r.turns,
                       "primary cause": autopsies[r.file].primary_cause,
                       "alarms": ", ".join(sorted(set(r.alarms))) or "–"} for r in failed])
st.dataframe(table, hide_index=True, width="stretch")
pick = st.selectbox("Open an incident", range(len(failed)),
                    format_func=lambda i: f"{failed[i].started[11:16]}Z · {failed[i].flight} {failed[i].route} · "
                                          f"{autopsies[failed[i].file].primary_cause}")
run, a = failed[pick], autopsies[failed[pick].file]

# ---------------------------------------------------------------- autopsy
with st.container(border=True):
    st.markdown(f"### {run.flight} · {run.route} · {run.date}")
    st.caption(f"{run.status.replace('_', ' ').title()} via {run.plan_source}" + (f" · {run.reason}" if run.reason else ""))
    st.markdown(f"**Deadlock point.** {a.deadlock_point}")
    st.markdown(f"**Where the domains collided.** {a.collision}")
    if a.causes:
        causes = pd.Series(a.causes).sort_values()
        fig = go.Figure(go.Bar(x=causes.values, y=causes.index, orientation="h", marker_color="#E0A458"))
        fig.update_layout(height=60 + 28 * len(causes), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="times this limit blocked a plan")
        st.plotly_chart(fig, width="stretch")

    s = scenario_for(run.dot_row, run.weather_mode) if run.dot_row is not None else None
    if s is None:
        st.info("This log has no DOT record reference, so the cost of failure can't be recomputed.")
    else:
        fin = plan_eval(s, run.final)
        best = plan_eval(s, a.best_ai_plan) if a.best_ai_plan else None
        st.markdown("**Cost of failure.** The plan that was executed vs the best plan the AI agents produced.")
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.markdown("**Executed (deterministic / override)**")
            st.markdown(fin["label"])
            st.metric("Cost", f"${fin['usd']:,.0f}" if fin["usd"] is not None else "–")
            st.metric("Holding fuel", f"{fin['fuel_kg']:,.0f} kg" if fin["fuel_kg"] is not None else "–",
                      f"{fin['fuel_kg'] * F.CO2_PER_KG_FUEL:,.0f} kg CO₂" if fin["fuel_kg"] is not None else None,
                      delta_color="off")
        with c2.container(border=True):
            st.markdown("**Best AI plan**" + ("" if a.best_ai_safe else " · ⚠️ never passed every hard check"))
            if best:
                st.markdown(best["label"])
                st.metric("Cost", f"${best['usd']:,.0f}" if best["usd"] is not None else "unsafe")
                st.metric("Holding fuel", f"{best['fuel_kg']:,.0f} kg" if best["fuel_kg"] is not None else "–",
                          f"{best['fuel_kg'] * F.CO2_PER_KG_FUEL:,.0f} kg CO₂" if best["fuel_kg"] is not None else None,
                          delta_color="off")
            else:
                st.markdown("The agents never put a plan on the table (they were unreachable).")

        # ------------------------------------------------------------ what-if replay
        st.markdown("**What-if replay.** Change something an airline or airport actually controls and replay the "
                    "same scenario. Safety minimums (final reserve, wake separation, crew duty) are never relaxed.")
        w1, w2, w3, w4 = st.columns(4)
        uplift = w1.slider("Extra fuel uplift (min of holding)", 0, 30, 10)
        staffing = w2.slider("Ramp staffing (%)", 50, 100, 100)
        spare = w3.toggle("One more stand free now", value=False)
        engine = w4.radio("Agents", ENGINES, index=1, help="Deterministic is free; Claude agents cost about $0.20.")
        if st.button("▶ Re-simulate", type="primary"):
            gates = [g.model_copy(update={"free_from": -999, "free_until": 10_000}) if spare and i == 0 and not g.remote
                     else g for i, g in enumerate(s.gates)]
            what_if = s.model_copy(update={
                "fob_at_eta_kg": s.fob_at_eta_kg + uplift * s.hold_burn_kg_per_min,
                "turnaround_min": int(round(s.turnaround_min * 100 / staffing)), "gates": gates})
            with st.spinner("Replaying the negotiation…"):
                out = negotiate(what_if, make_engine_agents(engine, "claude-opus-5-5", "medium"),
                                autonomy=Autonomy.AUTONOMOUS, save_log=False)
            new = plan_eval(what_if, out.proposal.model_dump())
            st.dataframe(pd.DataFrame([
                {"": "Original", "outcome": run.status, "plan": fin["label"],
                 "cost USD": fin["usd"], "holding fuel kg": fin["fuel_kg"]},
                {"": "What-if", "outcome": f"{out.status} ({out.plan_source})", "plan": new["label"],
                 "cost USD": new["usd"], "holding fuel kg": new["fuel_kg"]}]), hide_index=True, width="stretch")
            (st.success if out.status == "CONSENSUS" else st.warning)(
                "With these changes the agents converge." if out.status == "CONSENSUS" else
                f"Still no agreement: {F.autopsy(F.Run(**{**run.model_dump(), 'status': out.status})).primary_cause}"
                " remains the blocker. Try a different lever.")
            st.caption("What-if replays are not logged as operational decisions.")
