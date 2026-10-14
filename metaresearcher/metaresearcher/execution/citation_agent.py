"""Citation Agent: verifies claims in the report against sources."""
from __future__ import annotations

import json
import re

from ..llm_client import LLMClient
from ..models import LLMResponse
from ..prompts import CITATION_PROMPT


class CitationAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def verify(
        self, writer_output: str, researcher_output: str,
    ) -> tuple[list[dict], LLMResponse]:
        prompt = CITATION_PROMPT.format(
            writer_output=writer_output,
            researcher_output=researcher_output,
        )
        resp = self.llm.complete(prompt)
        return self._parse(resp.text), resp

    @staticmethod
    def _parse(text: str) -> list[dict]:
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(),
                         flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned).get("citations", [])
        except Exception:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0)).get("citations", [])
                except Exception:
                    pass
        return []
