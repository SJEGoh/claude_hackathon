"""Conversational layer: every agent can chat (DM or group) with its own memory and its own tools.

The bounded, structured negotiation (negotiation.py) is untouched; this is how a human talks to the team.
Each (agent, channel) pair keeps an append-only Claude message history, so DMs stay private to that agent.
Tools are deterministic code; agents never compute safety figures themselves.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Optional

import anthropic

from agents.base import MODELS_WITH_SERVER_FALLBACK
from constraints import VALIDATORS, Proposal
from monitoring import CallMetric, price
from scenario import Scenario

MAX_TOOL_STEPS = 8

CHAT_STYLE = """\
You are chatting with the duty officer (a human) in the airline/airport operations chat. Speak as the
specialist you are: short, precise, standard aviation terminology, local clock times (HH:MM). Use your
tools for any figure you quote; never guess a number. If something is outside your domain, say which
specialist owns it. A separate, bounded negotiation protocol with a deterministic safety engine decides
plans; in chat you advise, explain and check, you do not commit plans."""


def tool(name: str, description: str, props: dict, required: Optional[list] = None) -> dict:
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": props, "required": required or list(props),
                             "additionalProperties": False}}


PLAN_PROPS = {"landing_slot_min": {"type": "integer", "description": "minutes relative to SIBT"},
              "runway": {"type": "string"}, "gate": {"type": "string", "description": "stand id"}}


NO_FLIGHT = "No flight is loaded yet. Ask the Chief of Staff in #ops-team to load one."


def specialist_tools(domain: str, get_s: Callable[[], Optional[Scenario]]) -> tuple[list[dict], dict[str, Callable]]:
    """Deterministic tools a specialist may use in chat. Each only exposes that agent's own domain. The
    definitions never change (stable prompt prefix); they read the current flight when called."""
    defs = [tool("check_plan", f"Run the deterministic {domain} validator on a plan (your domain only).", PLAN_PROPS)]

    def check_plan(landing_slot_min: int, runway: str, gate: str) -> str:
        s = get_s()
        if s is None:
            return NO_FLIGHT
        r = VALIDATORS[domain](s, Proposal(landing_slot_min=landing_slot_min, runway=runway, gate=gate))
        return json.dumps({"hard_ok": r.hard_ok, "checks": [c.model_dump(exclude={"domain"}) for c in r.checks],
                           "metrics": r.metrics, "tldt_clock": s.clock(landing_slot_min)})
    impl: dict[str, Callable] = {"check_plan": check_plan}
    if domain == "ATC":
        from agents.atc_agent import ATCAgent
        defs.append(tool("free_slots", "AMAN conflict-free landing slots per runway (ATC checks only).", {}))

        def free_slots() -> str:
            s = get_s()
            return NO_FLIGHT if s is None else json.dumps(
                {k: [(t, s.clock(t)) for t in v] for k, v in ATCAgent.free_slots(s, per_runway=10).items()})
        impl["free_slots"] = free_slots
    if domain == "PILOT":
        defs.append(tool("fuel_at", "FMS fuel prediction (EFOB) for a landing slot.",
                         {"landing_slot_min": {"type": "integer"}}))

        def fuel_at(landing_slot_min: int) -> str:
            s = get_s()
            return NO_FLIGHT if s is None else json.dumps({
                "tldt": s.clock(landing_slot_min), "efob_kg": s.fuel_at(landing_slot_min),
                "min_landing_fuel_kg": s.min_landing_fuel_kg,
                "margin_min": round((s.fuel_at(landing_slot_min) - s.min_landing_fuel_kg) / s.hold_burn_kg_per_min, 1)})
        impl["fuel_at"] = fuel_at
    return defs, impl


class Conversation:
    """One agent's memory in one channel. Append-only history (required for preserved thinking)."""

    def __init__(self, agent_key: str, channel: str):
        self.agent_key, self.channel = agent_key, channel
        self.history: list = []
        self.group_cursor = 0  # how much of the group channel this agent has already been shown
        self.context_sig: Optional[str] = None  # which scenario's instruments it last received


def converse(client: anthropic.Anthropic, model: str, effort: str, system: str, conv: Conversation, content: str,
             tools: list[dict], impl: dict[str, Callable], on_tool: Optional[Callable[[str, dict, str], None]] = None,
             timeout_s: float = 120.0) -> tuple[str, list[CallMetric]]:
    """Send one user turn; run the tool loop (bounded); return the final text and per-call metrics."""
    start_len = len(conv.history)
    conv.history.append({"role": "user", "content": content})
    metrics: list[CallMetric] = []
    for _ in range(MAX_TOOL_STEPS):
        kwargs = dict(model=model, max_tokens=16000, system=system, messages=conv.history,
                      output_config={"effort": effort})
        if tools:
            kwargs["tools"] = tools
        if model in MODELS_WITH_SERVER_FALLBACK:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        started = time.monotonic()
        try:
            resp = client.with_options(timeout=timeout_s).beta.messages.create(**kwargs)
        except anthropic.APIError as e:
            del conv.history[start_len:]  # roll back so the history stays valid for the next message
            metrics.append(CallMetric(agent=conv.agent_key, turn=0, model=model,
                                      latency_s=round(time.monotonic() - started, 2), ok=False, error=str(e)[:200]))
            return f"⚠️ I couldn't reach the model ({type(e).__name__}). Please retry.", metrics
        u = resp.usage
        metrics.append(CallMetric(agent=conv.agent_key, turn=0, model=model,
                                  latency_s=round(time.monotonic() - started, 2), input_tokens=u.input_tokens,
                                  output_tokens=u.output_tokens,
                                  cost_usd=price(model, u.input_tokens, u.output_tokens)))
        conv.history.append({"role": "assistant", "content": resp.content})  # unchanged, thinking blocks included
        if resp.stop_reason == "refusal":
            return "I can't help with that request.", metrics
        uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not uses:
            text = "\n\n".join(b.text for b in resp.content if b.type == "text").strip()
            return text or "(no reply)", metrics
        results = []
        for b in uses:
            try:
                out = impl[b.name](**(b.input or {}))
                is_err = False
            except Exception as e:  # tool bugs go back to the model as errors, never crash the chat
                out, is_err = f"{type(e).__name__}: {e}", True
            if on_tool:
                on_tool(b.name, b.input or {}, out)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out), "is_error": is_err})
        conv.history.append({"role": "user", "content": results})
    return "I hit my tool-step limit for this message; ask me to continue.", metrics
