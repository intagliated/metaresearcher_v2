"""Regenerator: patches ONLY the agents mentioned in validation errors."""
from __future__ import annotations

from ..models import VALID_TOOLS, AgentDAG, ValidationError

_REQUIRED_CRITERIA = {
    "researcher": ["min_5_sources", "citations_present"],
    "analyst": ["numbers_validated"],
    "writer": ["word_count_500_2000"],
    "critic": ["gaps_identified"],
}


class Regenerator:
    def fix(
        self,
        dag: AgentDAG,
        errors: list[ValidationError],
    ) -> tuple[AgentDAG, list[str]]:
        log: list[str] = []
        victims: set[str] = set()

        for err in errors:
            if err.agent_name in dag.agents:
                victims.add(err.agent_name)
            log.append(f"[{err.check}] {err.agent_name}: {err.message}")

        for name in victims:
            spec = dag.agents[name]
            fixes: list[str] = []

            for tool in list(spec.tools):
                if tool not in VALID_TOOLS:
                    spec.tools.remove(tool)
                    fixes.append(f"removed tool '{tool}'")

            required = _REQUIRED_CRITERIA.get(spec.role, [])
            for crit in required:
                if crit not in spec.verification_criteria:
                    spec.verification_criteria.append(crit)
                    fixes.append(f"added criterion '{crit}'")

            if not spec.outputs:
                spec.outputs.append(f"{spec.role}_output")
                fixes.append("added default output")

            if fixes:
                spec.regenerated = True
                note = f"regenerated '{name}': " + "; ".join(fixes)
                spec.regeneration_history.append(note)
                log.append(note)

        return dag, log
