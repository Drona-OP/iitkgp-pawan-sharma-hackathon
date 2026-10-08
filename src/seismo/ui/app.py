"""Seismo dashboard: Risk Radar, Module A, Module B, Model Lab and Try It.

Run with `python -m seismo ui` (or `python -m seismo demo` to start the API and a replay as well).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Seismo", page_icon="📈", layout="wide")

from seismo.ui import common  # noqa: E402
from seismo.ui.pages import model_lab, module_a, module_b, radar, try_it  # noqa: E402

st.markdown(common.CSS, unsafe_allow_html=True)

controller = common.get_controller()
default_pack = common.settings.path("replay.default_pack")
if os.environ.get("SEISMO_AUTOSTART") == "1" and not controller.autostarted and default_pack.exists():
    controller.autostarted = True
    controller.start(default_pack, float(common.settings.get("replay.speed", 3600)))

pages = [
    st.Page(radar.render, title="Risk Radar", icon=":material/radar:", url_path="radar", default=True),
    st.Page(module_a.render, title="Module A - Rebalancer", icon=":material/stacked_line_chart:", url_path="module-a"),
    st.Page(module_b.render, title="Module B - Stress test", icon=":material/account_balance:", url_path="module-b"),
    st.Page(model_lab.render, title="Model Lab", icon=":material/science:", url_path="model-lab"),
    st.Page(try_it.render, title="Try It", icon=":material/edit_note:", url_path="try-it"),
]
nav = st.navigation(pages)
common.sidebar_replay()
nav.run()
