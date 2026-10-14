"""Single-agent baseline for comparison mode."""
from __future__ import annotations

from ..llm_client import LLMClient
from ..prompts import SINGLE_AGENT_PROMPT
from .metrics import MetricsTracker


class SingleAgentRunner:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, task: str) -> dict:
        metrics = MetricsTracker()
        prompt = SINGLE_AGENT_PROMPT.format(task=task)
        resp = self.llm.complete(prompt)
        metrics.record("single_agent", resp)
        return {
            "report": resp.text,
            "metrics": metrics,
        }
