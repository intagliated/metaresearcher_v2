"""Validator: 4 checks, plus a fault injector for chaos mode."""
from __future__ import annotations

import random

from ..models import VALID_ROLES, VALID_TOOLS, AgentDAG, ValidationError, ValidationResult

_REQUIRED_CRITERIA = {
    "researcher": {"min_5_sources", "citations_present"},
    "analyst": {"numbers_validated"},
    "writer": {"word_count_500_2000"},
    "critic": {"gaps_identified"},
}


def inject_fault(dag: AgentDAG, probability: float = 0.7) -> str | None:
    """Corrupt one agent spec so the normal checks have a real defect to catch."""
    if not dag.agents or random.random() >= probability:
        return None
    victim = random.choice(list(dag.agents.values()))
    removable = [c for c in victim.verification_criteria if c in _REQUIRED_CRITERIA.get(victim.role, set())]
    if removable and random.random() < 0.5:
        criterion = random.choice(removable)
        victim.verification_criteria.remove(criterion)
        return f"chaos: dropped criterion '{criterion}' from '{victim.name}'"
    victim.tools.append("quantum_search")
    return f"chaos: gave '{victim.name}' unknown tool 'quantum_search'"


class Validator:
    def validate(self, dag: AgentDAG) -> ValidationResult:
        errors: list[ValidationError] = []

        try:
            dag.topological_order()
        except ValueError as e:
            errors.append(ValidationError("__dag__", "structural", str(e)))

        if not dag.agents:
            errors.append(ValidationError("__dag__", "structural", "DAG has no agents"))
        if dag.agents and not dag.roots():
            errors.append(ValidationError("__dag__", "structural", "DAG has no root agent"))

        for name, spec in dag.agents.items():
            if spec.role not in VALID_ROLES:
                errors.append(ValidationError(name, "role", f"unknown role '{spec.role}'"))

        for name, spec in dag.agents.items():
            for inp in spec.inputs:
                if inp not in dag.agents:
                    errors.append(ValidationError(
                        name, "contract", f"input '{inp}' references unknown agent",
                    ))
            if not spec.outputs:
                errors.append(ValidationError(name, "contract", "agent declares no outputs"))

        for name, spec in dag.agents.items():
            for tool in spec.tools:
                if tool not in VALID_TOOLS:
                    errors.append(ValidationError(name, "tool", f"unknown tool '{tool}'"))

        for name, spec in dag.agents.items():
            expected = _REQUIRED_CRITERIA.get(spec.role, set())
            missing = expected - set(spec.verification_criteria)
            if missing:
                errors.append(ValidationError(
                    name, "criteria", f"missing verification criteria: {sorted(missing)}",
                ))

        return ValidationResult(passed=not errors, errors=errors)
