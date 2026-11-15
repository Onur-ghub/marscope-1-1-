"""LLM access for MARSCOPE, behind one small provider interface.

Providers
---------
* ``anthropic`` (default) - Claude via the official ``anthropic`` SDK.
  Key: ``ANTHROPIC_API_KEY`` (environment variable or Streamlit secret).
* ``openai_compatible`` - any server exposing ``/v1/chat/completions`` (uses ``requests``).
  Needs ``MARSCOPE_AI_BASE_URL``, ``MARSCOPE_AI_API_KEY`` and ``MARSCOPE_AI_MODEL``.

Select with ``MARSCOPE_AI_PROVIDER``. Keys are read server-side only and never sent to the
browser. Every failure is raised as :class:`AIUnavailable` with a readable message.
"""
from __future__ import annotations

import os

import config


class AIUnavailable(RuntimeError):
    """The AI service cannot answer (no key, network error, refusal, ...)."""


def setting(name: str, default: str | None = None) -> str | None:
    """Environment variable first, then Streamlit secrets (``.streamlit/secrets.toml``)."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:  # no secrets file, or not running under Streamlit
        return default


class AnthropicProvider:
    name = "Claude (Anthropic)"

    def __init__(self):
        self.model = setting("MARSCOPE_AI_MODEL", config.ANTHROPIC_MODEL)

    def available(self) -> tuple[bool, str]:
        if not (setting("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return False, "No ANTHROPIC_API_KEY set (environment variable or .streamlit/secrets.toml)."
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "The 'anthropic' package is not installed (pip install anthropic)."
        return True, ""

    def chat(self, system: str, messages: list[dict], max_tokens: int = 4000) -> str:
        import anthropic

        key = setting("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system, messages=messages,
                      output_config={"effort": "low"})
        try:
            try:
                # Server-side fallback lets a declined request be re-run on another model.
                resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                   fallbacks="default", **kwargs)
            except TypeError:  # older SDK without these parameters
                resp = client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise AIUnavailable("The Anthropic API key was rejected.") from exc
        except anthropic.RateLimitError as exc:
            raise AIUnavailable("Rate limited by the Anthropic API - try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise AIUnavailable(f"Anthropic API error {exc.status_code}.") from exc
        except anthropic.APIConnectionError as exc:
            raise AIUnavailable("Could not reach the Anthropic API.") from exc
        if resp.stop_reason == "refusal":
            raise AIUnavailable("The model declined to answer this request.")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if not text.strip():
            raise AIUnavailable("The model returned no text.")
        return text


class OpenAICompatibleProvider:
    name = "OpenAI-compatible endpoint"

    def __init__(self):
        self.base = (setting("MARSCOPE_AI_BASE_URL") or "").rstrip("/")
        self.model = setting("MARSCOPE_AI_MODEL")

    def available(self) -> tuple[bool, str]:
        missing = [n for n in ("MARSCOPE_AI_BASE_URL", "MARSCOPE_AI_API_KEY", "MARSCOPE_AI_MODEL") if not setting(n)]
        return (not missing), ("Missing " + ", ".join(missing) + "." if missing else "")

    def chat(self, system: str, messages: list[dict], max_tokens: int = 4000) -> str:
        import requests
        try:
            r = requests.post(f"{self.base}/v1/chat/completions", timeout=90,
                              headers={"Authorization": f"Bearer {setting('MARSCOPE_AI_API_KEY')}"},
                              json={"model": self.model, "max_tokens": max_tokens,
                                    "messages": [{"role": "system", "content": system}] + messages})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            raise AIUnavailable(f"The AI endpoint failed: {type(exc).__name__}.") from exc
        if not (text or "").strip():
            raise AIUnavailable("The model returned no text.")
        return text


PROVIDERS = {"anthropic": AnthropicProvider, "openai_compatible": OpenAICompatibleProvider}


def get_provider():
    key = (setting("MARSCOPE_AI_PROVIDER") or "anthropic").strip().lower()
    cls = PROVIDERS.get(key)
    if cls is None:
        raise AIUnavailable(f"Unknown MARSCOPE_AI_PROVIDER '{key}'. Use one of: {', '.join(PROVIDERS)}.")
    return cls()


def status() -> tuple[bool, str, str]:
    """(available, provider name, reason if unavailable)."""
    try:
        p = get_provider()
    except AIUnavailable as exc:
        return False, "none", str(exc)
    ok, why = p.available()
    return ok, p.name, why


def chat(system: str, messages: list[dict], max_tokens: int = 4000) -> str:
    p = get_provider()
    ok, why = p.available()
    if not ok:
        raise AIUnavailable(why)
    return p.chat(system, messages, max_tokens)
