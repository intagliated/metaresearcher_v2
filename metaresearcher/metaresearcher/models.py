"""Core data models for MetaResearcher."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["researcher", "analyst", "writer", "critic", "orchestrator"]

VALID_TOOLS = {"web_search", "calculator"}
VALID_ROLES = {"researcher", "analyst", "writer", "critic"}


@dataclass
class AgentSpec:
    name: str
    role: Role
    description: str
    tools: list[str] = field(default_factory=list)
    verification_criteria: list[str] = field(default_factory=list)
    prompt_template: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    regenerated: bool = False
    regeneration_history: list[str] = field(default_factory=list)


@dataclass
class AgentDAG:
    task: str
    agents: dict[str, AgentSpec] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def add_agent(self, spec: AgentSpec) -> None:
        self.agents[spec.name] = spec

    def add_edge(self, src: str, dst: str) -> None:
        if (src, dst) not in self.edges:
            self.edges.append((src, dst))

    def roots(self) -> list[str]:
        targets = {d for _, d in self.edges}
        return [n for n in self.agents if n not in targets]

    def topological_order(self) -> list[str]:
        indeg = {n: 0 for n in self.agents}
        for _, d in self.edges:
            indeg[d] += 1
        queue = [n for n, d in indeg.items() if d == 0]
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for s, d in self.edges:
                if s == n:
                    indeg[d] -= 1
                    if indeg[d] == 0:
                        queue.append(d)
        if len(order) != len(self.agents):
            raise ValueError("DAG contains a cycle")
        return order


@dataclass
class ValidationError:
    agent_name: str
    check: str
    message: str


@dataclass
class ValidationResult:
    passed: bool
    errors: list[ValidationError] = field(default_factory=list)


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    model: str


@dataclass
class AgentRunResult:
    agent_name: str
    role: Role
    output: str
    metrics: LLMResponse
    approved: bool | None = None


@dataclass
class ExecutionRecord:
    dag: AgentDAG
    results: dict[str, AgentRunResult] = field(default_factory=dict)
    report: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_s: float = 0.0
