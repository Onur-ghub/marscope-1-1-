"""Streamlit UI for the Astronaut Assistant tab (presentation only).

Mission logic lives in mission_planner, data in mission_context, language in assistant.
"""
from __future__ import annotations

import html

import streamlit as st

import config
from modules import assistant, mission_planner as mp

AVATARS = {"user": "🧑‍🚀", "assistant": "🛰️"}


def _esc(t) -> str:
    return html.escape(str(t))


def _recommendation(ctx: dict, evaluation, selected_key: str | None, badge, on_show_route, on_choose_dest):
    st.markdown("<div class='panel-title'>Mission recommendation</div>", unsafe_allow_html=True)
    dest = ctx.get("destination")
    if not dest or evaluation is None:
        st.info("No destination yet. Choose one on the start screen, in the sidebar, or by clicking the map "
                "in “Set destination” mode.")
        st.button("Choose a destination", on_click=on_choose_dest, type="primary")
        return
    rec = evaluation.recommended
    pos = ctx.get("astronaut_position") or {}
    if rec is None:
        st.error("No route meets the mission constraints. " + " ".join(
            f"Route {c.key}: {c.exclusions[0]}" for c in evaluation.excluded))
    else:
        m, cr = rec.raw, rec.criteria
        safety_txt = f"{cr['safety']:.0f}" if cr["safety"] is not None else "n/a"
        st.markdown(f"""<div class='ai-rec' style='border-left-color:{rec.color}'>
        <div class='ai-rec-dest'><span>Destination</span>{_esc(dest['name'])}
          <div class='coord'>{dest['lat']:.4f}° N · {dest['lon_east']:.4f}° E · from {_esc(pos.get('name', 'start'))}</div></div>
        <div class='ai-rec-route'>Recommended: <b>Route {rec.key} · {_esc(rec.name)}</b></div>
        <div class='mc-grid'>
          <div><span>{rec.score:.0f}</span>overall /100</div>
          <div><span>{m['distance_km']:.2f} km</span>distance</div>
          <div><span>{cr['science']:.0f}</span>science /100</div>
          <div><span>{_esc(safety_txt)}</span>safety /100 · {_esc(rec.safety_level)}</div>
        </div></div>""", unsafe_allow_html=True)
        lines = mp.explain_recommendation(evaluation)
        st.markdown("".join(f"<div class='why'>• {_esc(t)}</div>" for t in lines), unsafe_allow_html=True)
        if selected_key and selected_key != rec.key:
            st.button(f"Show Route {rec.key} on the map (currently Route {selected_key})", on_click=on_show_route,
                      args=(rec.key,))
        else:
            st.caption("The recommended route is the highlighted route on the Mission Map, with numbered stops.")

    rows = []
    for c in sorted(evaluation.candidates, key=lambda c: (c.rank is None, c.rank or 0)):
        if c.duplicate_of:
            rows.append(f"<tr class='excl'><td><b>{c.key}</b> {_esc(c.name)}</td>"
                        f"<td colspan='4'>same path as Route {c.duplicate_of}</td></tr>")
        elif c.eligible and c.score is not None:
            crit = c.criteria
            rows.append(f"<tr><td><b style='color:{c.color}'>{c.key}</b> {_esc(c.name)}</td><td>{c.score:.0f}</td>"
                        f"<td>{crit['science']:.0f}</td><td>{_esc(c.safety_level)}</td>"
                        f"<td>{c.raw['distance_km']:.2f}</td></tr>")
        else:
            rows.append(f"<tr class='excl'><td><b>{c.key}</b> {_esc(c.name)}</td>"
                        f"<td colspan='4'>excluded: {_esc(c.exclusions[0] if c.exclusions else '')}</td></tr>")
    st.markdown("<div class='panel-title'>All candidates</div><table class='ai-cands'><tr><th>Route</th>"
                "<th>Score</th><th>Science</th><th>Safety</th><th>km</th></tr>" + "".join(rows) + "</table>",
                unsafe_allow_html=True)
    w = evaluation.model.as_dict()
    st.caption(f"Scoring model: science {w['science']:.0%} · safety {w['safety']:.0%} · distance "
               f"{w['distance']:.0%} (set by Mission priorities in the sidebar). Deterministic: the AI explains "
               "these results but never computes them.")
    st.number_input("Maximum route length (km, 0 = no limit)", min_value=0.0, max_value=60.0, step=0.5,
                    key="max_route_km", help="Routes longer than this are excluded from the recommendation.")

    sci = ctx.get("destination_science") or {}
    st.markdown("<div class='panel-title'>At the destination</div>", unsafe_allow_html=True)
    for w_ in sci.get("why_interesting_marscope", []):
        st.markdown(f"<div class='why'>• {_esc(w_)}</div>", unsafe_allow_html=True)
    if sci.get("observations"):
        st.markdown(f"<div class='ai-sci'>{badge('observation')} " + "<br>".join(_esc(o) for o in sci["observations"][:3])
                    + "</div>", unsafe_allow_html=True)
    if sci.get("interpretations"):
        st.markdown(f"<div class='ai-sci'>{badge('literature')} Interpretation: " + "<br>".join(
            _esc(i) for i in sci["interpretations"][:2]) + "</div>", unsafe_allow_html=True)
    if not sci.get("observations"):
        st.markdown("<div class='ai-sci'>No published observations are catalogued within 3 km of this point.</div>",
                    unsafe_allow_html=True)
    with st.expander("Important uncertainties"):
        for u in ctx.get("uncertainties", []):
            st.markdown(f"- {u}")


def _chat(ctx: dict, sig: str, ai_status: tuple, evaluation=None):
    ok, provider, why = ai_status
    st.markdown("<div class='panel-title'>Mission assistant</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='ai-status {'on' if ok else 'off'}'>" + (
        f"● {_esc(provider)} · answers grounded in MARSCOPE mission data" if ok else
        "○ Offline mode: answers come straight from MARSCOPE data. Set ANTHROPIC_API_KEY for full "
        "conversation and general Mars questions.") + "</div>", unsafe_allow_html=True)

    chat = st.session_state.chat
    if st.session_state.get("_chat_sig") != sig:
        if chat and ctx.get("destination"):
            chat.append({"role": "event", "content": f"Mission updated · destination {ctx['destination']['name']}"
                         + (f" · recommended Route {ctx['route_evaluation']['recommended_route']}"
                            if ctx.get("route_evaluation") else "")})
        st.session_state._chat_sig = sig

    def ask(q):
        st.session_state.assistant_pending = q

    with st.container(key="ai_suggest"):
        for i, q in enumerate(assistant.suggested_questions(ctx)):
            st.button(q, key=f"sq_{i}", on_click=ask, args=(q,))

    box = st.container(height=470, key="ai_history")
    with box:
        if not chat:
            st.markdown("<div class='ai-empty'>Ask about your destination, the recommended route, the terrain, "
                        "or Mars in general. Follow-up questions keep the context.</div>", unsafe_allow_html=True)
        for msg in chat:
            if msg["role"] == "event":
                st.markdown(f"<div class='ai-event'>{_esc(msg['content'])}</div>", unsafe_allow_html=True)
                continue
            with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
                st.markdown(msg["content"])
                if msg.get("note"):
                    st.caption(msg["note"] + (" · sources: " + ", ".join(msg["sources"]) if msg.get("sources") else ""))

    typed = st.chat_input("Ask the assistant…", key="chat_input")
    question = typed or st.session_state.assistant_pending
    st.session_state.assistant_pending = None
    if not question:
        return
    with box:
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(question)
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            reply = get_reply(question, ctx, evaluation, sig, ok)
            st.markdown(reply.text)
            if reply.note:
                st.caption(reply.note + (" · sources: " + ", ".join(reply.sources) if reply.sources else ""))
    record(question, reply)


def get_reply(question: str, ctx: dict, evaluation, sig: str, ai_ok: bool) -> "assistant.Reply":
    """Answer (cached per mission state, so repeats cost no API call); never raises."""
    cache_key = (sig, question.strip().lower(), ai_ok)
    reply = st.session_state.chat_cache.get(cache_key)
    if reply is None:
        with st.spinner("Consulting mission data…"):
            try:
                reply = assistant.answer(question, st.session_state.chat, ctx, evaluation)
            except Exception as exc:  # never let the assistant break the app
                reply = assistant.Reply(f"The assistant hit an error and could not answer "
                                        f"({type(exc).__name__}). The rest of MARSCOPE is unaffected.", "error")
        if reply.mode != "error":
            st.session_state.chat_cache[cache_key] = reply
    return reply


def record(question: str, reply) -> None:
    """Add a question/answer pair to the shared conversation (Assistant tab and Mission Control)."""
    st.session_state.chat.append({"role": "user", "content": question})
    st.session_state.chat.append({"role": "assistant", "content": reply.text, "mode": reply.mode,
                                  "note": reply.note, "sources": reply.sources})


def render(ctx: dict, evaluation, selected_key, sig: str, ai_status: tuple, badge,
           on_show_route, on_choose_dest):
    left, right = st.columns([1.0, 1.25], gap="large")
    with left:
        _recommendation(ctx, evaluation, selected_key, badge, on_show_route, on_choose_dest)
    with right:
        _chat(ctx, sig, ai_status, evaluation)
        if st.session_state.chat:
            st.button("Clear conversation", key="chat_clear",
                      on_click=lambda: st.session_state.update(chat=[], chat_cache={}))
