"""Task Parser: LLM-backed decomposition with rule-based fallback."""
from __future__ import annotations

import json
import re

from ..llm_client import LLMClient
from ..prompts import TASK_PARSER_PROMPT


class TaskParser:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def parse(self, task: str) -> dict:
        prompt = TASK_PARSER_PROMPT.format(task=task)
        resp = self.llm.complete(prompt)
        parsed = self._try_json(resp.text)
        if self._is_well_formed(parsed):
            return parsed
        return self._fallback(task)

    @staticmethod
    def _is_well_formed(parsed: object) -> bool:
        if not isinstance(parsed, dict):
            return False
        agents, edges = parsed.get("agents"), parsed.get("edges")
        if not isinstance(agents, list) or not isinstance(edges, list):
            return False
        if not all(isinstance(a, dict) for a in agents):
            return False
        names = [a.get("name") for a in agents]
        if not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
            return False
        if not all(isinstance(a.get("role"), str) for a in agents):
            return False
        list_fields = ("tools", "verification_criteria", "inputs", "outputs")
        if not all(
            isinstance(a.get(f, []), list) and all(isinstance(x, str) for x in a.get(f, []))
            for a in agents for f in list_fields
        ):
            return False
        return all(
            isinstance(e, (list, tuple)) and len(e) == 2 and all(isinstance(x, str) for x in e)
            for e in edges
        )

    @staticmethod
    def _try_json(text: str) -> dict | None:
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
        return None

    @staticmethod
    def _fallback(task: str) -> dict:
        return {
            "agents": [
                {"name": "researcher", "role": "researcher",
                 "description": f"Gather sources on: {task}",
                 "tools": ["web_search"],
                 "verification_criteria": ["min_5_sources", "citations_present"],
                 "inputs": [], "outputs": ["sources", "raw_findings"]},
                {"name": "analyst", "role": "analyst",
                 "description": "Compute metrics and insights",
                 "tools": ["calculator"],
                 "verification_criteria": ["numbers_validated"],
                 "inputs": ["researcher"], "outputs": ["insights"]},
                {"name": "writer", "role": "writer",
                 "description": "Write the final report",
                 "tools": [],
                 "verification_criteria": ["word_count_500_2000"],
                 "inputs": ["researcher", "analyst"], "outputs": ["draft"]},
                {"name": "critic", "role": "critic",
                 "description": "Review the draft",
                 "tools": ["web_search"],
                 "verification_criteria": ["gaps_identified"],
                 "inputs": ["writer"], "outputs": ["review"]},
            ],
            "edges": [
                ["researcher", "analyst"],
                ["researcher", "writer"],
                ["analyst", "writer"],
                ["writer", "critic"],
            ],
        }
