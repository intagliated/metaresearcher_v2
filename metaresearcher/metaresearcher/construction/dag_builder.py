"""DAG Builder: converts parsed subtasks into an AgentDAG."""
from __future__ import annotations

from ..models import AgentDAG, AgentSpec
from ..prompts import (
    ANALYST_PROMPT, CRITIC_PROMPT, RESEARCHER_PROMPT, WRITER_PROMPT,
)

_PROMPT_BY_ROLE = {
    "researcher": RESEARCHER_PROMPT,
    "analyst": ANALYST_PROMPT,
    "writer": WRITER_PROMPT,
    "critic": CRITIC_PROMPT,
}


class DAGBuilder:
    def build(self, task: str, parsed: dict) -> AgentDAG:
        dag = AgentDAG(task=task)
        for raw in parsed["agents"]:
            role = raw["role"]
            spec = AgentSpec(
                name=raw["name"],
                role=role,
                description=raw.get("description", ""),
                tools=list(raw.get("tools", [])),
                verification_criteria=list(raw.get("verification_criteria", [])),
                prompt_template=_PROMPT_BY_ROLE.get(role, ""),
                inputs=list(raw.get("inputs", [])),
                outputs=list(raw.get("outputs", [])),
            )
            dag.add_agent(spec)

        for src, dst in parsed["edges"]:
            if src in dag.agents and dst in dag.agents:
                dag.add_edge(src, dst)

        # `inputs` feeds the prompts and `edges` orders the run; keep them one graph.
        # An input naming an unknown agent stays in `inputs` so the validator reports it.
        for name, spec in dag.agents.items():
            for inp in spec.inputs:
                if inp in dag.agents:
                    dag.add_edge(inp, name)
        for src, dst in dag.edges:
            if src not in dag.agents[dst].inputs:
                dag.agents[dst].inputs.append(src)

        return dag
