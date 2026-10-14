"""Prompt templates for all agents."""

TASK_PARSER_PROMPT = """You are a task decomposition engine.

Break the following research task into a DAG of agent subtasks.
Return ONLY valid JSON, no prose.

Task: {task}

Return this exact shape:
{{
  "agents": [
    {{"name": "researcher", "role": "researcher", "description": "...",
      "tools": ["web_search"], "verification_criteria": ["min_5_sources", "citations_present"],
      "inputs": [], "outputs": ["sources", "raw_findings"]}},
    {{"name": "analyst", "role": "analyst", "description": "...",
      "tools": ["calculator"], "verification_criteria": ["numbers_validated"],
      "inputs": ["researcher"], "outputs": ["insights"]}},
    {{"name": "writer", "role": "writer", "description": "...",
      "tools": [], "verification_criteria": ["word_count_500_2000"],
      "inputs": ["researcher", "analyst"], "outputs": ["draft"]}},
    {{"name": "critic", "role": "critic", "description": "...",
      "tools": ["web_search"], "verification_criteria": ["gaps_identified"],
      "inputs": ["writer"], "outputs": ["review"]}}
  ],
  "edges": [["researcher","analyst"],["researcher","writer"],["analyst","writer"],["writer","critic"]]
}}
"""

RESEARCHER_PROMPT = """You are a research agent.

Task: {task}

Use web search to gather at least 5 credible sources on this topic.
For each finding, record: claim, source URL, date.

Output format (markdown):
## Sources
1. [claim] - [url] ([date])
2. ...

## Raw Findings
- ...
"""

ANALYST_PROMPT = """You are a data analyst agent.

Task: {task}

Researcher findings:
{researcher_output}

Compute growth rates, comparisons, and 3-5 key insights.
Show your calculations. Only use numbers present in the findings.
"""

WRITER_PROMPT = """You are a report writer agent.

Task: {task}

Researcher findings:
{researcher_output}

Analyst insights:
{analyst_output}

{critic_feedback}

Write a clear market report between 500 and 2000 words.
Include an executive summary, key findings, and a conclusion.
Cite sources inline as [n].
"""

CRITIC_PROMPT = """You are a critic agent.

Task: {task}

Draft report:
{writer_output}

Review the draft. Check for: unsupported claims, missing sources,
weak analysis, structural problems.

Return ONLY valid JSON:
{{
  "approved": true|false,
  "score": 0-100,
  "issues": ["..."],
  "feedback": "Specific instructions for the writer."
}}
"""

CITATION_PROMPT = """You are a citation verification agent.

Report:
{writer_output}

Researcher sources:
{researcher_output}

For every factual claim in the report, verify it against the sources.
Return ONLY valid JSON:
{{
  "citations": [
    {{"claim": "...", "source": "...", "verified": true|false, "note": "..."}}
  ],
  "verified_count": 0,
  "unverified_count": 0
}}
"""

SINGLE_AGENT_PROMPT = """You are a single-agent research assistant.

Task: {task}

Research this topic and write a complete market report (500-2000 words).
Include sources and analysis. Do everything in one pass.
"""
