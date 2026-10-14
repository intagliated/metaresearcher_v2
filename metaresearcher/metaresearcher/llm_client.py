"""Auto-detecting LLM client: Anthropic -> OpenAI -> Mock."""
from __future__ import annotations

import hashlib
import os
import time

from .models import LLMResponse

# A 2000-word report is about 2,700 tokens; 2048 would cut the writer off.
ANTHROPIC_MAX_TOKENS = 4096


class LLMClient:
    """Tries Anthropic, then OpenAI, then falls back to mock mode."""

    def __init__(self) -> None:
        self.provider: str = "mock"
        self._client = None
        self._model = "mock-model"

        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self._client = anthropic.Anthropic()
                self.provider = "anthropic"
                self._model = "claude-sonnet-4-20250514"
            except Exception:
                pass

        if self.provider == "mock" and os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                self._client = openai.OpenAI()
                self.provider = "openai"
                self._model = "gpt-4o-mini"
            except Exception:
                pass

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        start = time.time()

        if self.provider == "anthropic":
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=ANTHROPIC_MAX_TOKENS,
                system=system or "You are a helpful research assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            return LLMResponse(
                text=msg.content[0].text,
                tokens_in=msg.usage.input_tokens,
                tokens_out=msg.usage.output_tokens,
                latency_s=time.time() - start,
                model=self._model,
            )

        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful research assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            usage = resp.usage
            return LLMResponse(
                text=resp.choices[0].message.content or "",
                tokens_in=usage.prompt_tokens if usage else 0,
                tokens_out=usage.completion_tokens if usage else 0,
                latency_s=time.time() - start,
                model=self._model,
            )

        return self._mock_complete(prompt, start)

    def _mock_complete(self, prompt: str, start: float) -> LLMResponse:
        seed = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        text = self._mock_text(prompt, seed)
        tokens_in = max(50, len(prompt) // 4)
        tokens_out = max(80, len(text) // 4)
        time.sleep(0.15)
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=time.time() - start,
            model="mock-model",
        )

    @staticmethod
    def _mock_text(prompt: str, seed: str) -> str:
        p = prompt.lower()

        if "task decomposition" in p:
            return (
                '{"agents": ['
                '{"name":"researcher","role":"researcher","description":"Gathers sources",'
                '"tools":["web_search"],"verification_criteria":["min_5_sources","citations_present"],'
                '"inputs":[],"outputs":["sources","raw_findings"]},'
                '{"name":"analyst","role":"analyst","description":"Analyzes data",'
                '"tools":["calculator"],"verification_criteria":["numbers_validated"],'
                '"inputs":["researcher"],"outputs":["insights"]},'
                '{"name":"writer","role":"writer","description":"Writes report",'
                '"tools":[],"verification_criteria":["word_count_500_2000"],'
                '"inputs":["researcher","analyst"],"outputs":["draft"]},'
                '{"name":"critic","role":"critic","description":"Reviews draft",'
                '"tools":["web_search"],"verification_criteria":["gaps_identified"],'
                '"inputs":["writer"],"outputs":["review"]}'
                '], "edges":[["researcher","analyst"],["researcher","writer"],'
                '["analyst","writer"],["writer","critic"]]}'
            )

        if "citation verification" in p:
            return (
                '{"citations": ['
                '{"claim":"Market grew 12% YoY","source":"https://example.com/a","verified":true,"note":"matches source"},'
                '{"claim":"Top 3 brands hold 60% share","source":"https://example.com/b","verified":true,"note":"matches source"},'
                '{"claim":"Regulation expected in 2027","source":"https://example.com/c","verified":false,"note":"not in sources"}'
                '], "verified_count": 2, "unverified_count": 1}'
            )

        if "critic agent" in p or '"approved"' in p:
            if "## revision notes" in p:
                return (
                    '{"approved": true, "score": 88, '
                    '"issues": [], '
                    '"feedback": "Feedback addressed: growth math shown, conclusion extended."}'
                )
            return (
                '{"approved": false, "score": 72, '
                '"issues": ["Two claims lack sources", "Growth rate calculation unclear", "Conclusion is thin"], '
                '"feedback": "Add citations to claims 3 and 5. Show the growth-rate math. Expand the conclusion with a 2026 outlook."}'
            )

        if "research agent" in p:
            return (
                "## Sources\n"
                "1. Global e-scooter market reached $42B in 2025 - https://example.com/a (2025-08)\n"
                "2. Segway leads with 24% share - https://example.com/b (2025-07)\n"
                "3. Regulatory framework expected 2027 - https://example.com/c (2025-06)\n"
                "4. Battery costs fell 18% YoY - https://example.com/d (2025-05)\n"
                "5. APAC is fastest-growing region at 22% CAGR - https://example.com/e (2025-04)\n\n"
                "## Raw Findings\n"
                "- Market size: $42B (2025)\n"
                "- YoY growth: 12%\n"
                "- Top 3 brands: Segway, Xiaomi, Bird (60% combined)\n"
                "- Battery cost trend: -18% YoY\n"
                "- APAC CAGR: 22%\n"
            )

        if "data analyst" in p:
            return (
                "## Calculations\n"
                "- Market growth: (42 - 37.5) / 37.5 = 12.0% YoY\n"
                "- APAC share: 22% CAGR -> projected 2027 size = 42 * 1.22^2 = $62.5B\n"
                "- Top-3 concentration: 60% (high oligopoly signal)\n\n"
                "## Insights\n"
                "1. Market is growing steadily at 12% YoY, led by APAC at 22% CAGR.\n"
                "2. Top-3 brand concentration (60%) suggests an oligopoly forming.\n"
                "3. Falling battery costs (-18%) are driving margin expansion.\n"
                "4. Regulation in 2027 is a key risk to watch.\n"
                "5. Projected 2027 market: ~$62.5B if current trends hold.\n"
            )

        if "report writer" in p:
            report = (
                "# Electric Scooter Market Report\n\n"
                "## Executive Summary\n"
                "The global electric scooter market reached $42B in 2025, growing 12% YoY [1]. "
                "APAC leads growth at a 22% CAGR [5], while falling battery costs (-18%) [4] "
                "are expanding margins across the sector.\n\n"
                "## Key Findings\n"
                "**Market size and growth.** The market hit $42B in 2025, up 12% from $37.5B [1].\n\n"
                "**Competitive landscape.** Segway, Xiaomi, and Bird together hold ~60% share [2], "
                "signaling an emerging oligopoly.\n\n"
                "**Regional dynamics.** APAC is the fastest-growing region at 22% CAGR [5]; "
                "if sustained, the market could reach ~$62.5B by 2027.\n\n"
                "**Cost tailwinds.** Battery costs fell 18% YoY [4], improving unit economics.\n\n"
                "**Regulatory risk.** A formal regulatory framework is expected in 2027 [3].\n\n"
                "## Conclusion\n"
                "The e-scooter market is in a healthy growth phase driven by APAC demand and "
                "falling input costs. Investors should watch 2027 regulation and oligopoly "
                "consolidation as the two defining dynamics of the next cycle.\n"
            )
            if "previous critic feedback" in p:
                report += (
                    "\n## Revision Notes\n"
                    "Growth math: (42 - 37.5) / 37.5 = 12.0% YoY [1]. "
                    "2026 outlook: continued double-digit growth, with APAC above 20% CAGR [5]. "
                    "Claims 3 and 5 now cite [3] and [5].\n"
                )
            return report

        return f"[mock output for seed {seed}] " + prompt[:200]
