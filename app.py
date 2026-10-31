"""A-CDM multi-agent operations.   streamlit run app.py

Pages: Ops chat (team group chat + a private DM with each agent) and Control room (dashboard, weather
provenance, monitoring, batch evaluation, EDA).
"""
import streamlit as st

st.set_page_config(page_title="A-CDM Multi-Agent Ops", page_icon="✈️", layout="wide")

from ui_common import apply_theme, customise_dialog  # noqa: E402  (after set_page_config)

apply_theme()
if st.sidebar.button("🎨 Customise agents", width="stretch"):
    customise_dialog()
page = st.navigation([
    st.Page("views/ops_chat.py", title="Ops chat", icon="💬", default=True),
    st.Page("views/control_room.py", title="Control room", icon="🛫"),
])
page.run()
