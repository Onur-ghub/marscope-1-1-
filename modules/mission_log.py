"""Mission activity timeline: records real actions taken in this MARSCOPE session.

Only events the application actually performs are logged (with UTC timestamps).
Simulation steps are logged too, and tagged as simulated.
"""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st


def log(event: str, detail: str = "", kind: str = "app") -> None:
    events = st.session_state.setdefault("mission_events", [])
    if events and events[-1]["event"] == event and events[-1]["detail"] == detail:
        return  # same action re-rendered, not a new event
    events.append({"utc": datetime.now(timezone.utc), "event": event, "detail": detail, "kind": kind})


def events() -> list[dict]:
    return st.session_state.get("mission_events", [])


def mission_elapsed_s() -> float | None:
    ev = events()
    if not ev:
        return None
    return (datetime.now(timezone.utc) - ev[0]["utc"]).total_seconds()
