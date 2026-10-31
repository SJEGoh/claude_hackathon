"""Ops settings: the knobs an ops room actually turns, wired into the safety checks, agents and weather watch."""
import pandas as pd
import streamlit as st

import ops_settings as O
import watch
from escalation import push
from ui_common import fire_test_alert, hero

hero("Ops settings · runway condition, approach mode, operating limits, ground resources, weather watch and "
     "notifications. Every value is read live by the safety checks and the agents.", compact=True)

cfg = O.get()
tab_cfg, tab_watch = st.tabs(["⚙️ Settings", "📡 Weather watch board"])

with tab_cfg:
    with st.form("ops_settings"):
        st.subheader("🛬 Runway and approach")
        c1, c2 = st.columns(2)
        rwycc = c1.select_slider("Runway condition code (ICAO GRF RWYCC)", options=list(range(6, -1, -1)),
                                 value=cfg.rwycc, format_func=lambda v: O.RWYCC_LABEL[v],
                                 help="Lower codes reduce the crosswind limit and add spacing; 0 closes the runway. "
                                      "Illustrative table: replace with your operator's figures in ops_settings.py.")
        approach = c2.selectbox("Approach mode", ["AUTO", "VISUAL", "ILS_CAT_I", "ILS_CAT_II_III"],
                                index=["AUTO", "VISUAL", "ILS_CAT_I", "ILS_CAT_II_III"].index(cfg.approach),
                                help="AUTO follows the METAR (IFR/LIFR adds spacing). CAT II/III = low-visibility "
                                     "procedures: +2 min on every arrival pair.")
        c1, c2 = st.columns(2)
        xw = c1.number_input("Max crosswind incl. gust (kt)", 5.0, 45.0, float(cfg.max_crosswind_kt))
        tw = c2.number_input("Max tailwind (kt)", 0.0, 15.0, float(cfg.max_tailwind_kt))

        st.subheader("🧑‍✈️ Flight crew limits")
        c1, c2, c3 = st.columns(3)
        adv = c1.number_input("MINIMUM FUEL advisory margin (min)", 0, 30, cfg.min_fuel_advisory_margin_min)
        sev = c2.number_input("Max holding in SEV turbulence/icing (min)", 0, 30, cfg.severe_hold_limit_min)
        mct = c3.number_input("Minimum connection time (min)", 20, 120, cfg.min_connect_time_min)

        st.subheader("🧳 Ground resources")
        c1, c2, c3 = st.columns(3)
        staff = c1.slider("Ramp staffing (%)", 25, 100, cfg.ramp_staffing_pct,
                          help="Below 100% stretches every turnaround proportionally.")
        ot = c2.number_input("Crew overtime legal cap (min)", 0, 120, cfg.max_crew_overtime_min)
        remote = c3.toggle("Allow remote stands", cfg.allow_remote_stands)

        st.subheader("🤝 Negotiation")
        c1, c2 = st.columns(2)
        budget = c1.number_input("Time budget per negotiation (s)", 10, 600, cfg.time_budget_s)
        autonomy = c2.selectbox("Default autonomy", ["AUTONOMOUS", "SUPERVISED", "STEP_BY_STEP"],
                                index=["AUTONOMOUS", "SUPERVISED", "STEP_BY_STEP"].index(cfg.default_autonomy))

        st.subheader("📡 Weather watch and notifications")
        airports = st.text_input("Watchlist (IATA codes, comma separated)", ", ".join(cfg.watch_airports))
        c1, c2, c3 = st.columns(3)
        interval = c1.number_input("Check every (s)", 60, 3600, cfg.watch_interval_s, step=60)
        gust = c2.number_input("Gust alert (kt)", 10, 60, cfg.gust_alert_kt)
        rain = c3.number_input("Heavy rain (mm/h)", 1.0, 50.0, float(cfg.heavy_rain_mm_h))
        checks = {c: st.checkbox(c.replace("_", " ").title(), cfg.checks.get(c, True), key=f"chk_{c}")
                  for c in O.CHECKS}
        c1, c2, c3 = st.columns(3)
        push_sev = c1.selectbox("Push me from severity", ["INFO", "WARNING", "ALARM"],
                                index=["INFO", "WARNING", "ALARM"].index(cfg.push_min_severity))
        sound = c2.toggle("Sound on ALARM", cfg.notify_sound)
        desktop = c3.toggle("Mac desktop notifications", cfg.notify_desktop)
        ntfy = st.text_input("Phone push topic (ntfy)", cfg.ntfy_topic, placeholder="e.g. acdm-ops-stevson-7f3k",
                             help="Install the free 'ntfy' app, subscribe to this topic name, and alerts arrive as "
                                  "phone notifications. Pick a long, unguessable name: topics are public.")
        if st.form_submit_button("💾 Save settings", type="primary"):
            O.save(O.OpsSettings(
                rwycc=rwycc, approach=approach, max_crosswind_kt=xw, max_tailwind_kt=tw,
                min_fuel_advisory_margin_min=adv, severe_hold_limit_min=sev, min_connect_time_min=mct,
                ramp_staffing_pct=staff, allow_remote_stands=remote, max_crew_overtime_min=ot, time_budget_s=budget,
                default_autonomy=autonomy, watch_airports=[a.strip().upper() for a in airports.split(",") if a.strip()],
                watch_interval_s=interval, checks=checks, gust_alert_kt=gust, heavy_rain_mm_h=rain,
                push_min_severity=push_sev, notify_sound=sound, notify_desktop=desktop, ntfy_topic=ntfy.strip()))
            st.success("Saved. The safety checks, agents and weather watch use the new values immediately.")

    cur = O.get()
    st.info(f"**Effective now:** crosswind limit {cur.crosswind_limit():.0f} kt at RWYCC {cur.rwycc} · "
            f"+{cur.extra_spacing(False)} min spacing in VMC / +{cur.extra_spacing(True)} in IMC · turnaround ×"
            f"{100 / cur.ramp_staffing_pct:.2f}" + (" · **RUNWAYS CLOSED (RWYCC 0)**" if cur.runway_closed_by_condition() else ""))
    c1, c2 = st.columns(2)
    if c1.button("🧪 Fire a test ALARM (sound + every channel)"):
        sent = fire_test_alert()
        st.success("Test fired. Sent to: " + (", ".join(sent) or "this screen only: add an ntfy topic, a webhook, "
                                                               "or turn on Mac notifications above"))
    if c2.button("↺ Restore defaults"):
        O.save(O.OpsSettings())
        st.rerun()

with tab_watch:
    c1, c2 = st.columns([1, 3])
    if c1.button("🔄 Check now", type="primary"):
        with st.spinner("Checking METAR, TAF, PIREPs, SIGMETs and forecasts…"):
            evs = watch.poll()
        st.toast(f"{len(evs)} change(s)")
    c2.caption(f"Watching {', '.join(O.get().watch_airports)} every {O.get().watch_interval_s // 60} min. "
               "You are notified only when an alert appears, escalates or clears.")
    alerts = watch.active_alerts()
    if alerts:
        icon = {"ALARM": "🔴", "WARNING": "🟠", "INFO": "🔵"}
        st.dataframe(pd.DataFrame([{"": icon[a.severity], "airport": a.airport, "check": a.check, "alert": a.title,
                                    "detail": a.detail, "observed": a.observed} for a in alerts]),
                     hide_index=True, width="stretch")
    else:
        st.success("No active alerts on the watchlist.")
    st.markdown("**Recent changes**")
    ev = watch.recent_events(40)
    if ev:
        st.dataframe(pd.DataFrame([{"at": e.at[11:16] + "Z", "change": e.kind, "severity": e.alert.severity,
                                    "airport": e.alert.airport, "alert": e.alert.title,
                                    "sent to": ", ".join(e.channels)} for e in ev]), hide_index=True, width="stretch")
