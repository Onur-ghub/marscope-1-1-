"""Streamlit version compatibility (requirements: streamlit>=1.50).

Newer keyword arguments (for example ``segmented_control(required=True)``, added after 1.50) raise
``TypeError`` on older installs. ``install()`` wraps the widget/layout functions once at start-up:
any keyword the installed version does not know is dropped, with the closest older equivalent where
one exists:

* ``required=True`` on segmented_control / pills: emulated. When the user clicks the active option
  (which would clear it), the previous value is restored.
* ``width="stretch"`` on buttons, charts and tables: mapped to ``use_container_width=True`` (also for
  ``plotly_chart``, which takes ``**kwargs`` on older versions and would otherwise warn).
* ``horizontal=`` / alignment on containers: dropped (the container stacks vertically).

Nothing changes on versions that already support the arguments.
"""
from __future__ import annotations

import functools
import inspect

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

NAMES = ("segmented_control", "pills", "button", "download_button", "link_button", "form_submit_button",
         "container", "columns", "selectbox", "multiselect", "slider", "select_slider", "radio", "checkbox",
         "toggle", "text_input", "text_area", "number_input", "dataframe", "data_editor", "plotly_chart",
         "image", "expander", "popover", "tabs", "metric", "markdown", "caption", "chat_input", "toast",
         "file_uploader", "date_input", "time_input", "color_picker", "altair_chart", "pydeck_chart")
_DROPPED: set[str] = set()
_installed = False


def _params(fn) -> tuple[set[str], bool] | None:
    """Named parameters, and whether the function also takes **kwargs."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    return ({n for n, p in sig.parameters.items() if p.kind != p.VAR_KEYWORD},
            any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()))


def _required_callback(key: str, on_change, args, kwargs):
    """on_change replacement that refuses to clear a required single-select widget."""
    prev_key = f"_compat_prev_{key}"

    def cb():
        ss = st.session_state
        if ss.get(key) is None and ss.get(prev_key) is not None:
            ss[key] = ss[prev_key]
            return
        ss[prev_key] = ss.get(key)
        if on_change:
            on_change(*(args or ()), **(kwargs or {}))
    return cb


def _wrap(fn, name: str):
    found = _params(fn)
    if found is None or getattr(fn, "_compat", False):
        return fn
    known, varkw = found
    if varkw and not ("width" not in known and "use_container_width" in known):
        return fn               # takes **kwargs and nothing to translate (e.g. old plotly_chart only needs width)

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        unknown = [k for k in kw if k not in known]
        if varkw:               # only translate width=; other keywords pass through as before
            unknown = [k for k in unknown if k == "width"]
        if unknown:
            if "required" in unknown and kw.get("required") and name in ("segmented_control", "pills") \
                    and kw.get("key"):
                key = kw["key"]
                st.session_state.setdefault(f"_compat_prev_{key}", st.session_state.get(key))
                kw["on_change"] = _required_callback(key, kw.get("on_change"), kw.pop("args", None),
                                                     kw.pop("kwargs", None))
            if "width" in unknown and kw.get("width") == "stretch" and "use_container_width" in known:
                kw["use_container_width"] = True
            for k in unknown:
                kw.pop(k, None)
                _DROPPED.add(f"{name}({k})")
        return fn(*a, **kw)

    wrapper._compat = True
    return wrapper


def install() -> None:
    """Patch st.<name> and DeltaGenerator.<name> once; a no-op where nothing is missing."""
    global _installed
    if _installed:
        return
    _installed = True
    for name in NAMES:
        if hasattr(DeltaGenerator, name):
            setattr(DeltaGenerator, name, _wrap(getattr(DeltaGenerator, name), name))
        if hasattr(st, name):
            setattr(st, name, _wrap(getattr(st, name), name))


def dropped() -> list[str]:
    """Keyword arguments ignored on this Streamlit version (for Data & Methods / debugging)."""
    return sorted(_DROPPED)
