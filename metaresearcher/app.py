"""MetaResearcher Streamlit UI."""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from metaresearcher.construction.dag_builder import DAGBuilder
from metaresearcher.construction.regenerator import Regenerator
from metaresearcher.construction.task_parser import TaskParser
from metaresearcher.construction.validator import Validator, inject_fault
from metaresearcher.execution.agents import CriticAgent
from metaresearcher.execution.orchestrator import Orchestrator
from metaresearcher.execution.single_agent import SingleAgentRunner
from metaresearcher.graph import render_dag
from metaresearcher.llm_client import LLMClient

MAX_ATTEMPTS = 3
DEFAULT_TASK = "Write a market report on electric scooters"

st.set_page_config(page_title="MetaResearcher", layout="wide")


def _init_state() -> None:
    defaults = {
        "dag": None,
        "validation": None,
        "regeneration_log": [],
        "execution": None,
        "single_agent": None,
        "executed_agents": set(),
        "writer_feedback": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

st.sidebar.title("MetaResearcher")
task = st.sidebar.text_area("Research task", value=DEFAULT_TASK, height=80)
chaos_mode = st.sidebar.toggle(
    "Chaos Mode",
    value=False,
    help="Inject random validation failures to demo targeted regeneration.",
)
run_build = st.sidebar.button("Build & Validate DAG", width="stretch")

llm = LLMClient()
st.sidebar.caption(f"LLM provider: **{llm.provider}** ({llm._model})")

if run_build:
    with st.spinner("Parsing task..."):
        parser = TaskParser(llm)
        parsed = parser.parse(task)

    builder = DAGBuilder()
    dag = builder.build(task, parsed)

    validator = Validator()
    regenerator = Regenerator()
    regen_log: list[str] = []
    if chaos_mode:
        fault = inject_fault(dag)
        if fault:
            regen_log.append(fault)
    validation = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        validation = validator.validate(dag)
        if validation.passed:
            regen_log.append(f"Attempt {attempt}: passed")
            break
        regen_log.append(f"Attempt {attempt}: {len(validation.errors)} error(s)")
        dag, fixes = regenerator.fix(dag, validation.errors)
        regen_log.extend(f"  -> {line}" for line in fixes)

    st.session_state["dag"] = dag
    st.session_state["validation"] = validation
    st.session_state["regeneration_log"] = regen_log
    st.session_state["execution"] = None
    st.session_state["executed_agents"] = set()
    st.session_state["writer_feedback"] = ""

tab_build, tab_exec, tab_compare, tab_report = st.tabs(
    ["Construction", "Execution", "Comparison", "Report"]
)

with tab_build:
    dag = st.session_state["dag"]
    validation = st.session_state["validation"]

    if dag is None:
        st.info("Enter a task in the sidebar and click Build & Validate DAG.")
    else:
        regenerated = {n for n, s in dag.agents.items() if s.regenerated}
        col_dag, col_status = st.columns([2, 1])

        with col_dag:
            st.subheader("Agent DAG")
            st.graphviz_chart(render_dag(dag, regenerated=regenerated))

        with col_status:
            st.subheader("Validation")
            if validation and validation.passed:
                st.success(f"Passed ({len(dag.agents)} agents)")
            else:
                st.error("Failed")
                if validation:
                    for err in validation.errors:
                        st.write(f"- **{err.agent_name}** [{err.check}]: {err.message}")
            st.subheader("Regeneration history")
            if st.session_state["regeneration_log"]:
                for line in st.session_state["regeneration_log"]:
                    st.text(line)
            else:
                st.caption("No regeneration needed.")

        st.subheader("Agent specifications")
        rows = [{
            "name": s.name,
            "role": s.role,
            "tools": ", ".join(s.tools) or "-",
            "criteria": ", ".join(s.verification_criteria) or "-",
            "inputs": ", ".join(s.inputs) or "-",
            "outputs": ", ".join(s.outputs) or "-",
            "regenerated": "yes" if s.regenerated else "",
        } for s in dag.agents.values()]
        st.dataframe(pd.DataFrame(rows), width="stretch")

        st.divider()
        can_execute = bool(validation and validation.passed)
        if st.button("Execute DAG", disabled=not can_execute,
                     type="primary", width="stretch"):
            orch = Orchestrator(llm)
            with st.spinner("Executing multi-agent pipeline..."):
                exec_data = orch.run(dag)
            st.session_state["execution"] = exec_data
            st.session_state["executed_agents"] = set(exec_data["results"].keys())
            st.success("Execution complete. See the Execution tab.")
        if not can_execute:
            st.caption("Execute is disabled until validation passes.")

with tab_exec:
    dag = st.session_state["dag"]
    execution = st.session_state["execution"]
    if execution is None:
        st.info("Run the DAG from the Construction tab first.")
    else:
        executed = st.session_state["executed_agents"]
        regenerated = {n for n, s in dag.agents.items() if s.regenerated}
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("Execution DAG")
            st.graphviz_chart(render_dag(dag, executed=executed,
                                         regenerated=regenerated))
            st.subheader("Execution log")
            log_box = st.container(height=260)
            for line in execution["log"]:
                log_box.text(f"[{time.strftime('%H:%M:%S')}] {line}")
        with col_right:
            st.subheader("Agent outputs")
            for name, res in execution["results"].items():
                with st.expander(f"{name} ({res.role})"):
                    st.markdown(res.output[:4000])
                    st.caption(
                        f"tokens_in={res.metrics.tokens_in} | "
                        f"tokens_out={res.metrics.tokens_out} | "
                        f"latency={res.metrics.latency_s:.2f}s | "
                        f"model={res.metrics.model}"
                    )

        critic = next((r for r in execution["results"].values() if r.role == "critic"), None)
        if critic:
            st.divider()
            st.subheader("Critic review")
            review = CriticAgent.parse_review(critic.output)
            c1, c2, c3 = st.columns(3)
            c1.metric("Approved", "yes" if review.get("approved") else "no")
            c2.metric("Score", review.get("score", "-"))
            c3.metric("Issues", len(review.get("issues", [])))
            if review.get("issues"):
                st.write("**Issues:**")
                for issue in review["issues"]:
                    st.write(f"- {issue}")
            if review.get("feedback"):
                st.info(review["feedback"])
            if not review.get("approved"):
                if st.button("Send back to Writer", type="primary"):
                    writer_feedback = review.get("feedback", "")
                    prior = execution["outputs"]
                    orch = Orchestrator(llm)
                    with st.spinner("Re-running Writer with critic feedback..."):
                        new_exec = orch.run(
                            dag,
                            critic_feedback=writer_feedback,
                            only_agents={n for n, s in dag.agents.items()
                                         if s.role in ("writer", "critic")},
                            prior_outputs=prior,
                        )
                    new_exec["metrics"].per_agent = (
                        execution["metrics"].per_agent
                        + new_exec["metrics"].per_agent
                    )
                    new_exec["results"] = {**execution["results"], **new_exec["results"]}
                    new_exec["log"] = execution["log"] + new_exec["log"]
                    st.session_state["execution"] = new_exec
                    st.session_state["executed_agents"] = set(new_exec["results"])
                    st.session_state["writer_feedback"] = writer_feedback
                    st.rerun()

        st.divider()
        st.subheader("Metrics")
        metrics = execution["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tokens in", metrics.total_tokens_in)
        m2.metric("Tokens out", metrics.total_tokens_out)
        m3.metric("Total tokens", metrics.total_tokens)
        m4.metric("Latency", f"{metrics.total_latency_s:.2f}s")
        if metrics.per_agent:
            fig = go.Figure(data=[
                go.Bar(
                    x=[m.agent for m in metrics.per_agent],
                    y=[m.tokens_in + m.tokens_out for m in metrics.per_agent],
                    marker_color="#3498DB",
                )
            ])
            fig.update_layout(
                title="Tokens per agent",
                xaxis_title="Agent", yaxis_title="Tokens",
                height=320, margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, width="stretch")

        st.divider()
        st.subheader("Citation verification")
        if execution["citations"]:
            for c in execution["citations"]:
                badge = "OK" if c.get("verified") else "!!"
                st.write(f"[{badge}] **{c.get('claim','')}** - {c.get('source','')}")
                if c.get("note"):
                    st.caption(c["note"])
        else:
            st.caption("No citations parsed.")

with tab_compare:
    dag = st.session_state["dag"]
    execution = st.session_state["execution"]
    if dag is None:
        st.info("Build a DAG first.")
    else:
        st.subheader("Single-agent vs multi-agent")
        if st.button("Run single-agent baseline", width="stretch"):
            runner = SingleAgentRunner(llm)
            with st.spinner("Running single-agent baseline..."):
                st.session_state["single_agent"] = runner.run(dag.task)
        single = st.session_state["single_agent"]
        if single and execution:
            ma_tokens = execution["metrics"].total_tokens
            sa_tokens = single["metrics"].total_tokens
            ma_lat = execution["metrics"].total_latency_s
            sa_lat = single["metrics"].total_latency_s
            c1, c2, c3 = st.columns(3)
            c1.metric("Multi-agent tokens", ma_tokens)
            c2.metric("Single-agent tokens", sa_tokens)
            c3.metric("Token ratio", f"{ma_tokens / max(sa_tokens,1):.1f}x")
            c4, c5, c6 = st.columns(3)
            c4.metric("Multi-agent latency", f"{ma_lat:.2f}s")
            c5.metric("Single-agent latency", f"{sa_lat:.2f}s")
            c6.metric("Latency ratio", f"{ma_lat / max(sa_lat,0.01):.1f}x")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### Multi-agent report")
                st.markdown(execution["report"][:3000])
            with col_b:
                st.markdown("### Single-agent report")
                st.markdown(single["report"][:3000])
        elif single is None:
            st.caption("Click above to run the baseline.")
        else:
            st.caption("Execute the multi-agent DAG first.")

with tab_report:
    execution = st.session_state["execution"]
    if execution is None:
        st.info("Run the DAG to produce a report.")
    else:
        report = execution["report"]
        st.markdown(report)
        st.download_button(
            "Download report (.txt)",
            data=report,
            file_name="metaresearcher_report.txt",
            mime="text/plain",
        )
        with st.expander("Raw execution JSON"):
            raw = {
                "task": st.session_state["dag"].task,
                "agents": {
                    n: {
                        "role": r.role,
                        "output": r.output,
                        "tokens_in": r.metrics.tokens_in,
                        "tokens_out": r.metrics.tokens_out,
                        "latency_s": r.metrics.latency_s,
                        "model": r.metrics.model,
                    }
                    for n, r in execution["results"].items()
                },
                "citations": execution["citations"],
                "totals": {
                    "tokens_in": execution["metrics"].total_tokens_in,
                    "tokens_out": execution["metrics"].total_tokens_out,
                    "latency_s": execution["metrics"].total_latency_s,
                },
            }
            st.json(raw)
