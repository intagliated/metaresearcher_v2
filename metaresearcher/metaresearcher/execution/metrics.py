"""Metrics tracker for token/latency accounting."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models import LLMResponse


@dataclass
class AgentMetric:
    agent: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_s: float


@dataclass
class MetricsTracker:
    per_agent: list[AgentMetric] = field(default_factory=list)

    def record(self, agent_name: str, resp: LLMResponse) -> None:
        self.per_agent.append(AgentMetric(
            agent=agent_name,
            model=resp.model,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_s=resp.latency_s,
        ))

    @property
    def total_tokens_in(self) -> int:
        return sum(m.tokens_in for m in self.per_agent)

    @property
    def total_tokens_out(self) -> int:
        return sum(m.tokens_out for m in self.per_agent)

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_in + self.total_tokens_out

    @property
    def total_latency_s(self) -> float:
        return sum(m.latency_s for m in self.per_agent)
