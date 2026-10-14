import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from metaresearcher.construction.dag_builder import DAGBuilder
from metaresearcher.construction.task_parser import TaskParser
from metaresearcher.llm_client import LLMClient
from metaresearcher.models import LLMResponse

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class _StubLLM:
    def __init__(self, text):
        self.text = text

    def complete(self, prompt, system=""):
        return LLMResponse(self.text, 1, 1, 0.0, "stub")


def _built_app():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.button[0].click().run()
    return at


def _executed_app():
    at = _built_app()
    next(b for b in at.button if b.label == "Execute DAG").click().run()
    return at


def test_chaos_mode_fault_is_repaired_and_validation_passes(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.toggle[0].set_value(True)
    at.sidebar.button[0].click().run()

    assert [s.value for s in at.success] == ["Passed (4 agents)"]
    history = [t.value for t in at.text]
    assert any("regenerated '" in line for line in history), history


def test_send_back_to_writer_keeps_earlier_agents_and_log():
    at = _executed_app()
    next(b for b in at.button if b.label == "Send back to Writer").click().run()

    execution = at.session_state["execution"]
    assert list(execution["results"]) == ["researcher", "analyst", "writer", "critic"]
    assert "run researcher (researcher)" in execution["log"]
    assert "skip researcher (cached)" in execution["log"]
    assert at.session_state["executed_agents"] == {"researcher", "analyst", "writer", "critic"}


def test_api_error_is_raised_not_replaced_by_mock_output(monkeypatch):
    import anthropic

    class Failing:
        def __init__(self, *a, **k):
            self.messages = self

        def create(self, **kwargs):
            raise RuntimeError("rate limited")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(anthropic, "Anthropic", Failing)

    llm = LLMClient()
    assert llm.provider == "anthropic"
    with pytest.raises(RuntimeError, match="rate limited"):
        llm.complete("anything")
    assert llm.provider == "anthropic"


@pytest.mark.parametrize("bad_plan", [
    '{"agents": [{"name": "a"}], "edges": []}',
    '{"agents": [{"role": "writer"}], "edges": []}',
    '{"agents": ["researcher"], "edges": []}',
    '{"agents": [{"name": "a", "role": "writer"}], "edges": [["a"]]}',
])
def test_malformed_llm_plan_falls_back_to_a_buildable_plan(bad_plan):
    parsed = TaskParser(_StubLLM(bad_plan)).parse("some task")
    dag = DAGBuilder().build("some task", parsed)
    assert list(dag.agents) == ["researcher", "analyst", "writer", "critic"]


def test_well_formed_llm_plan_is_kept():
    plan = (
        '{"agents": [{"name": "solo", "role": "writer"}], "edges": []}'
    )
    parsed = TaskParser(_StubLLM(plan)).parse("some task")
    assert [a["name"] for a in parsed["agents"]] == ["solo"]


def _agent(name, role, inputs=(), criteria=(), tools=()):
    return {
        "name": name, "role": role, "description": "d", "tools": list(tools),
        "verification_criteria": list(criteria), "inputs": list(inputs), "outputs": ["o"],
    }


def _build(plan):
    import json
    parsed = TaskParser(_StubLLM(json.dumps(plan))).parse("some task")
    return DAGBuilder().build("some task", parsed)


class _RecordingLLM:
    """Numbers each researcher reply so a downstream prompt shows which ones it received."""

    def __init__(self):
        self.prompts = []
        self.researcher_calls = 0

    def complete(self, prompt, system=""):
        self.prompts.append(prompt)
        if "research agent" in prompt.lower():
            self.researcher_calls += 1
            return LLMResponse(f"FINDINGS-{self.researcher_calls}", 1, 1, 0.0, "stub")
        return LLMResponse("ok", 1, 1, 0.0, "stub")


def test_report_and_citations_do_not_depend_on_agent_names():
    from metaresearcher.execution.orchestrator import Orchestrator
    dag = _build({
        "agents": [
            _agent("scout", "researcher"),
            _agent("author", "writer", ["scout"]),
        ],
        "edges": [["scout", "author"]],
    })
    out = Orchestrator(LLMClient()).run(dag)
    assert out["report"].startswith("# Electric Scooter Market Report")
    assert len(out["citations"]) == 3


def test_inputs_without_edges_still_order_upstream_first():
    dag = _build({
        "agents": [_agent("writer", "writer", ["researcher"]), _agent("researcher", "researcher")],
        "edges": [],
    })
    assert dag.topological_order() == ["researcher", "writer"]


def test_edges_without_inputs_still_feed_the_downstream_agent():
    dag = _build({
        "agents": [_agent("researcher", "researcher"), _agent("writer", "writer")],
        "edges": [["researcher", "writer"]],
    })
    assert dag.agents["writer"].inputs == ["researcher"]


def test_unknown_role_fails_validation():
    from metaresearcher.construction.validator import Validator
    dag = _build({"agents": [_agent("s", "summariser")], "edges": []})
    result = Validator().validate(dag)
    assert not result.passed
    assert [(e.agent_name, e.check) for e in result.errors if e.check == "role"] == [("s", "role")]


def test_two_upstreams_with_the_same_role_both_reach_the_downstream_agent():
    from metaresearcher.execution.orchestrator import Orchestrator
    llm = _RecordingLLM()
    dag = _build({
        "agents": [
            _agent("r1", "researcher"), _agent("r2", "researcher"),
            _agent("writer", "writer", ["r1", "r2"]),
        ],
        "edges": [["r1", "writer"], ["r2", "writer"]],
    })
    Orchestrator(llm).run(dag)
    writer_prompt = next(p for p in llm.prompts if "report writer" in p.lower())
    assert "FINDINGS-1" in writer_prompt and "FINDINGS-2" in writer_prompt


@pytest.mark.parametrize("bad_plan", [
    '{"agents": [{"name": "w", "role": "writer", "inputs": "researcher"}], "edges": []}',
    '{"agents": [{"name": "w", "role": "writer", "tools": null}], "edges": []}',
    '{"agents": [{"name": "a", "role": "writer"}, {"name": "a", "role": "critic"}], "edges": []}',
    '{"agents": [{"name": "a", "role": "writer"}], "edges": [["a", ["x"]]]}',
])
def test_plan_with_wrong_field_types_falls_back_instead_of_crashing(bad_plan):
    parsed = TaskParser(_StubLLM(bad_plan)).parse("some task")
    dag = DAGBuilder().build("some task", parsed)
    assert list(dag.agents) == ["researcher", "analyst", "writer", "critic"]


def test_critic_reply_that_is_not_an_object_gets_the_default_review():
    from metaresearcher.execution.agents import CriticAgent
    assert CriticAgent.parse_review("[1]") == CriticAgent.parse_review("not json")


def test_send_back_to_writer_works_when_agents_are_not_named_after_roles(monkeypatch):
    import json
    names = {"researcher": "scout", "analyst": "number_cruncher", "writer": "author", "critic": "reviewer"}
    original = LLMClient._mock_text

    def renamed(prompt, seed):
        text = original(prompt, seed)
        if "task decomposition" not in prompt.lower():
            return text
        plan = json.loads(text)
        for a in plan["agents"]:
            a["name"] = names[a["name"]]
            a["inputs"] = [names[i] for i in a["inputs"]]
        plan["edges"] = [[names[s], names[d]] for s, d in plan["edges"]]
        return json.dumps(plan)

    monkeypatch.setattr(LLMClient, "_mock_text", staticmethod(renamed))
    at = _executed_app()
    assert not at.exception
    assert "Critic review" in [s.value for s in at.subheader]
    next(b for b in at.button if b.label == "Send back to Writer").click().run()

    assert not at.exception
    execution = at.session_state["execution"]
    assert list(execution["results"]) == ["scout", "number_cruncher", "author", "reviewer"]
    assert "skip scout (cached)" in execution["log"]


def test_send_back_to_writer_ends_with_critic_approval_in_mock_mode():
    at = _executed_app()
    first = at.session_state["execution"]
    assert "Revision Notes" not in first["report"]
    assert '"approved": false' in first["results"]["critic"].output

    next(b for b in at.button if b.label == "Send back to Writer").click().run()

    execution = at.session_state["execution"]
    assert "## Revision Notes" in execution["report"]
    assert '"approved": true' in execution["results"]["critic"].output
    assert "Send back to Writer" not in [b.label for b in at.button]
