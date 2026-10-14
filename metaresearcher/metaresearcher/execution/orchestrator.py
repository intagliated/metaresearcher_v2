"""Orchestrator: runs the verified DAG."""
from __future__ import annotations

from ..llm_client import LLMClient
from ..models import AgentDAG, AgentRunResult
from .agents import AGENT_CLASS_BY_ROLE, CriticAgent
from .citation_agent import CitationAgent
from .metrics import MetricsTracker


class Orchestrator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.metrics = MetricsTracker()
        self.log: list[str] = []

    def _log(self, msg: str) -> None:
        self.log.append(msg)

    def run(
        self,
        dag: AgentDAG,
        critic_feedback: str = "",
        only_agents: set[str] | None = None,
        prior_outputs: dict[str, str] | None = None,
    ) -> dict:
        outputs: dict[str, str] = dict(prior_outputs or {})
        results: dict[str, AgentRunResult] = {}

        order = dag.topological_order()
        self._log(f"Execution order: {' -> '.join(order)}")

        for name in order:
            spec = dag.agents[name]

            if only_agents is not None and name not in only_agents:
                self._log(f"skip {name} (cached)")
                continue

            cls = AGENT_CLASS_BY_ROLE.get(spec.role)
            if cls is None:
                self._log(f"skip {name}: no implementation for role {spec.role}")
                continue

            agent = cls(spec, self.llm)
            self._log(f"run {name} ({spec.role})")

            kwargs: dict = {"task": dag.task}
            by_role: dict[str, list[str]] = {}
            for upstream in spec.inputs:
                up_spec = dag.agents.get(upstream)
                if up_spec and upstream in outputs:
                    by_role.setdefault(up_spec.role, []).append(outputs[upstream])
            for role, texts in by_role.items():
                kwargs[f"{role}_output"] = "\n\n".join(texts)

            if spec.role == "writer" and critic_feedback:
                kwargs["critic_feedback"] = critic_feedback

            text, resp = agent.run(**kwargs)
            outputs[name] = text
            self.metrics.record(name, resp)

            approved = None
            if spec.role == "critic":
                review = CriticAgent.parse_review(text)
                approved = bool(review.get("approved", True))

            results[name] = AgentRunResult(
                agent_name=name, role=spec.role,
                output=text, metrics=resp, approved=approved,
            )

        writers = [n for n in order if dag.agents[n].role == "writer" and n in outputs]
        researchers = [n for n in order if dag.agents[n].role == "researcher" and n in outputs]

        report = outputs[writers[-1]] if writers else ""
        if not report:
            report = self._synthesize(dag, outputs)

        citations: list[dict] = []
        if writers and researchers:
            self._log("verify citations")
            cit_agent = CitationAgent(self.llm)
            citations, cit_resp = cit_agent.verify(
                writer_output=outputs[writers[-1]],
                researcher_output="\n\n".join(outputs[n] for n in researchers),
            )
            self.metrics.record("citation", cit_resp)

        return {
            "results": results,
            "outputs": outputs,
            "report": report,
            "citations": citations,
            "log": self.log,
            "metrics": self.metrics,
        }

    @staticmethod
    def _synthesize(dag: AgentDAG, outputs: dict[str, str]) -> str:
        parts = [f"# Report: {dag.task}\n"]
        for name, text in outputs.items():
            parts.append(f"\n## {name}\n{text}\n")
        return "\n".join(parts)
