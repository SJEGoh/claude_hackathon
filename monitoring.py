"""Per-call latency / token / cost telemetry for every Claude call."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

# USD per million tokens (input, output). Output includes thinking tokens, which are billed as output.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class CallMetric(BaseModel):
    agent: str
    turn: int
    model: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    ok: bool = True
    error: Optional[str] = None


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 6)


def summarise(metrics: list[CallMetric]) -> dict:
    ok = [m for m in metrics if m.ok]
    lat = sorted(m.latency_s for m in ok)
    return {
        "calls": len(metrics), "failed_calls": len(metrics) - len(ok),
        "input_tokens": sum(m.input_tokens for m in metrics), "output_tokens": sum(m.output_tokens for m in metrics),
        "cost_usd": round(sum(m.cost_usd for m in metrics), 4),
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_max_s": lat[-1] if lat else None,
        "by_agent": {a: round(sum(m.cost_usd for m in metrics if m.agent == a), 4)
                     for a in sorted({m.agent for m in metrics})},
    }
