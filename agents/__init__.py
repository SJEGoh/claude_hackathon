"""The three independent domain specialists plus the Chief of Staff moderator, one module each."""
from __future__ import annotations

import os
from typing import Optional

import anthropic

import config  # noqa: F401  (loads .env)
from agents.atc_agent import ATCAgent
from agents.base import (
    DEFAULT_MODEL, DEFAULT_TIMEOUT_S, AgentError, AgentMessage, AgentReply, BaseAgent, ConstraintClaim, StakeholderVote,
    render_message, render_report, reply_schema,
)
from agents.ground_agent import GroundAgent
from agents.moderator import ModeratorAgent, ModeratorMessage, render_brief
from agents.pilot_agent import PilotAgent

__all__ = ["ATCAgent", "PilotAgent", "GroundAgent", "ModeratorAgent", "BaseAgent", "AgentError", "AgentMessage",
           "AgentReply", "ConstraintClaim", "ModeratorMessage", "StakeholderVote", "DEFAULT_MODEL", "DEFAULT_TIMEOUT_S",
           "make_agents", "render_message", "render_brief", "render_report", "reply_schema"]

ROLES = {"ATC": ATCAgent, "PILOT": PilotAgent, "GROUND": GroundAgent, "MODERATOR": ModeratorAgent}


def make_agents(api_key: Optional[str] = None, model: str = DEFAULT_MODEL, effort: str = "low",
                moderator: bool = True, models: Optional[dict[str, str]] = None,
                timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, BaseAgent]:
    """models: optional per-role override, e.g. {"MODERATOR": "claude-haiku-5-5"}."""
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), max_retries=1)
    roles = [r for r in ROLES if moderator or r != "MODERATOR"]
    return {r: ROLES[r](client, (models or {}).get(r, model), effort, timeout_s) for r in roles}
