"""Air Control Management Center.   streamlit run app.py

Pages: CHATS (team group chat + a private DM with each agent, voice input), CONTROLS (dashboard, weather,
monitoring) and Ops settings (limits layer, weather watch, notifications, guided demo).
Sidebar on every page: MAYDAY (with confirmation) at the top; weather watch, alert sounds and agent customisation
at the bottom.
"""
import time

import streamlit as st

st.set_page_config(page_title="Air Control Management Center", page_icon="✈️", layout="wide")

import ops_settings  # noqa: E402  (after set_page_config)
import watch  # noqa: E402
from ui_common import apply_theme, customise_dialog, fire_test_alert, mayday_dialog, play_alert  # noqa: E402

apply_theme()

# --- top of the sidebar, every page: MAYDAY -------------------------------------------------------------
if st.sidebar.button("🚨 DECLARE MAYDAY", type="primary", width="stretch",
                     help="Opens a confirmation. On CONFIRM the negotiation is abandoned and the deterministic "
                          "emergency plan goes to every desk at once."):
    mayday_dialog()

if st.session_state.pop("play_unlock", False):
    play_alert(force=True)  # confirmation chime: proves sound works and unlocks audio for later alerts
    st.toast("Alert sounds on", icon="🔔")


@st.fragment(run_every=60)
def weather_watch() -> None:
    """Automated checks: polls the watchlist every `watch_interval_s`, alerts only on change."""
    cfg = ops_settings.get()
    last = st.session_state.get("watch_last", 0.0)
    if time.time() - last >= cfg.watch_interval_s:
        st.session_state.watch_last = time.time()
        try:
            events = watch.poll()
        except Exception as e:  # noqa: BLE001 - the watch must never break the app
            events = []
            st.toast(f"Weather watch error: {type(e).__name__}", icon="⚠️")
        for ev in events:
            icon = "✅" if ev.kind == "CLEARED" else {"ALARM": "🚨", "WARNING": "⚠️", "INFO": "ℹ️"}[ev.alert.severity]
            st.toast(f"{ev.kind}: {ev.alert.airport} {ev.alert.title}", icon=icon)
        if cfg.notify_sound and any(e.kind != "CLEARED" and e.alert.severity == "ALARM" for e in events):
            play_alert()
    alerts = watch.active_alerts()
    alarms = sum(a.severity == "ALARM" for a in alerts)
    label = (f"🔴 {alarms} ALARM · {len(alerts)} active" if alarms else
             f"🟠 {len(alerts)} active" if alerts else "🟢 all clear")
    with st.expander(f"📡 Weather watch · {label}", expanded=False):
        for a in alerts[:8]:
            st.caption(f"{ {'ALARM': '🔴', 'WARNING': '🟠', 'INFO': '🔵'}[a.severity]} **{a.airport}** {a.title}")
        nxt = max(0, int(cfg.watch_interval_s - (time.time() - st.session_state.get("watch_last", 0))))
        st.caption(f"Next check in ~{nxt // 60}m{nxt % 60:02d}s · settings on the Ops settings page")
        if st.button("🧪 Test alert (sound + notifications)", key="watch_test", width="stretch",
                     help="Browsers only play sound after you've clicked on the page once; this click counts."):
            sent = fire_test_alert()
            st.caption("Sent to: " + (", ".join(sent) or "this screen only (add a phone/webhook in Ops settings)"))


page = st.navigation([
    st.Page("views/ops_chat.py", title="CHATS", icon="💬", default=True),
    st.Page("views/control_room.py", title="CONTROLS", icon="🛫"),
    st.Page("views/failure_analysis.py", title="FAILURE ANALYSIS", icon="🔍"),
    st.Page("views/ops_settings_page.py", title="OPS SETTINGS", icon="⚙️"),
])
page.run()

# --- bottom of the sidebar, every page -------------------------------------------------------------------
with st.sidebar:
    st.divider()
    weather_watch()

    def _sound_toggled() -> None:
        st.session_state.play_unlock = st.session_state.sound_enabled

    st.toggle("🔔 Alert sounds", key="sound_enabled", on_change=_sound_toggled,
              help="Browsers only allow sound after you click on the page: turning this on is that click. "
                   "Plays a chime for ALARMs, emergencies and tests; warnings stay silent.")
    if st.button("🎨 Customise agents", width="stretch"):
        customise_dialog()
