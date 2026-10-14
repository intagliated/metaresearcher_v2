"""Graphviz rendering of the AgentDAG."""
from __future__ import annotations

from graphviz import Digraph

from .models import AgentDAG

_ROLE_COLORS = {
    "researcher": "#AED6F1",
    "analyst":    "#A9DFBF",
    "writer":     "#F9E79F",
    "critic":     "#F5B7B1",
    "orchestrator": "#D7BDE2",
}


def render_dag(
    dag: AgentDAG,
    executed: set[str] | None = None,
    regenerated: set[str] | None = None,
) -> Digraph:
    executed = executed or set()
    regenerated = regenerated or set()

    g = Digraph("metaresearcher")
    g.attr(rankdir="LR", bgcolor="transparent")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")

    for name, spec in dag.agents.items():
        fill = _ROLE_COLORS.get(spec.role, "#EEEEEE")

        if name in executed:
            penwidth = "3"
            color = "#27AE60"
        elif name in regenerated:
            penwidth = "3"
            color = "#E67E22"
        else:
            penwidth = "1"
            color = "#34495E"

        label = f"{name}\n({spec.role})\ntools: {','.join(spec.tools) or '-'}"
        g.node(name, label=label, fillcolor=fill,
               color=color, penwidth=penwidth)

    for src, dst in dag.edges:
        g.edge(src, dst)

    return g
