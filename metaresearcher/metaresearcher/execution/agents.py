"""Concrete agent implementations."""
from __future__ import annotations

import json
import re

from ..llm_client import LLMClient
from ..models import AgentSpec, LLMResponse
from ..prompts import (
    ANALYST_PROMPT, CRITIC_PROMPT, RESEARCHER_PROMPT, WRITER_PROMPT,
)


class BaseAgent:
    def __init__(self, spec: AgentSpec, llm: LLMClient) -> None:
        self.spec = spec
        self.llm = llm

    def run(self, **kwargs) -> tuple[str, LLMResponse]:
        raise NotImplementedError


class ResearcherAgent(BaseAgent):
    def run(self, **kwargs) -> tuple[str, LLMResponse]:
        prompt = RESEARCHER_PROMPT.format(task=kwargs["task"])
        resp = self.llm.complete(prompt)
        return resp.text, resp


class AnalystAgent(BaseAgent):
    def run(self, **kwargs) -> tuple[str, LLMResponse]:
        prompt = ANALYST_PROMPT.format(
            task=kwargs["task"],
            researcher_output=kwargs.get("researcher_output", ""),
        )
        resp = self.llm.complete(prompt)
        return resp.text, resp


class WriterAgent(BaseAgent):
    def run(self, **kwargs) -> tuple[str, LLMResponse]:
        feedback = kwargs.get("critic_feedback", "")
        feedback_block = f"## Previous critic feedback\n{feedback}\n" if feedback else ""
        prompt = WRITER_PROMPT.format(
            task=kwargs["task"],
            researcher_output=kwargs.get("researcher_output", ""),
            analyst_output=kwargs.get("analyst_output", ""),
            critic_feedback=feedback_block,
        )
        resp = self.llm.complete(prompt)
        return resp.text, resp


class CriticAgent(BaseAgent):
    def run(self, **kwargs) -> tuple[str, LLMResponse]:
        prompt = CRITIC_PROMPT.format(
            task=kwargs["task"],
            writer_output=kwargs.get("writer_output", ""),
        )
        resp = self.llm.complete(prompt)
        return resp.text, resp

    @staticmethod
    def parse_review(text: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(),
                         flags=re.MULTILINE).strip()
        candidates = [cleaned]
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            candidates.append(m.group(0))
        for candidate in candidates:
            try:
                review = json.loads(candidate)
            except Exception:
                continue
            if isinstance(review, dict):
                return review
        return {"approved": True, "score": 80, "issues": [], "feedback": ""}


AGENT_CLASS_BY_ROLE = {
    "researcher": ResearcherAgent,
    "analyst": AnalystAgent,
    "writer": WriterAgent,
    "critic": CriticAgent,
}
