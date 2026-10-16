"""Thesis-quality visualization helpers for the ATTICA causal-analysis project.

Design goals
------------
* compact figures that remain readable after LaTeX scaling;
* consistent node placement across domain-level figures;
* neutral edges so causal direction is visually dominant;
* vector PDF output for the thesis, plus optional PNG for notebooks;
* no query-specific labels such as "confounder" or "mediator" on generic graphs.

The graph routines use Graphviz through ``pygraphviz`` when available because
Graphviz produces cleaner layered DAG layouts and spline routing than a large
manual matplotlib canvas. A clear ImportError is raised when it is unavailable.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    import pygraphviz as pgv
except ImportError:  # pragma: no cover - depends on local environment
    pgv = None


# -----------------------------------------------------------------------------
# GLOBAL THESIS STYLE
# -----------------------------------------------------------------------------
DOMAIN_ORDER = [
    "Demographics",
    "SES",
    "Behavioral",
    "Treatment",
    "Obesity",
    "Metabolic",
    "Biomarkers",
    "Conditions",
    "Outcomes",
]

# Layout stages are visual only; they do not add causal assumptions.
DOMAIN_VISUAL_STAGES = [
    ["Demographics", "SES"],
    ["Behavioral", "Treatment"],
    ["Obesity"],
    ["Metabolic", "Biomarkers"],
    ["Conditions"],
    ["Outcomes"],
]

DOMAIN_FILL = {
    "Demographics": "#EDF2F7",
    "SES": "#EDF2F7",
    "Behavioral": "#F4F1EA",
    "Treatment": "#EAF3ED",
    "Obesity": "#EDF2F7",
    "Metabolic": "#EDF2F7",
    "Biomarkers": "#EDF2F7",
    "Conditions": "#F5EFE7",
    "Outcomes": "#F6E9E9",
}

DISPLAY_LABELS = {
    "Demographics": "Demographics",
    "SES": "SES",
    "Behavioral": "Behavioral",
    "Treatment": "Treatment",
    "Obesity": "Obesity",
    "Metabolic": "Metabolic",
    "Biomarkers": "Biomarkers",
    "Conditions": "Conditions",
    "Outcomes": "Outcomes",
}

EDGE_COLOR = "#64748B"
NODE_BORDER = "#526072"


def _require_graphviz():
    if pgv is None:
        raise ImportError(
            "Thesis-quality graph rendering requires pygraphviz + Graphviz. "
            "Install Graphviz and pygraphviz, then rerun the notebook."
        )


def _normalise_output_path(filename):
    if filename is None:
        return None
    path = Path(filename)
    if path.suffix.lower() not in {".pdf", ".png", ".svg"}:
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _draw_graphviz(graph, filename=None, show=True, also_png=True):
    """Render a Graphviz graph compactly and return saved file paths."""
    _require_graphviz()
    graph.layout("dot")

    saved = []
    out = _normalise_output_path(filename)

    if out is not None:
        graph.draw(str(out))
        saved.append(out)

        # For thesis use PDF; for notebook inspection keep a PNG sibling.
        if also_png and out.suffix.lower() != ".png":
            png = out.with_suffix(".png")
            graph.draw(str(png))
            saved.append(png)
        elif out.suffix.lower() == ".png":
            pdf = out.with_suffix(".pdf")
            graph.draw(str(pdf))
            saved.append(pdf)

    if show:
        from IPython.display import Image, display
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            preview = Path(tmp.name)
        graph.draw(str(preview))
        display(Image(filename=str(preview)))
        preview.unlink(missing_ok=True)

    return saved


def _base_domain_graph(directed=True, title=None, show_title=False):
    """Create the common compact ATTICA domain graph canvas."""
    _require_graphviz()
    A = pgv.AGraph(strict=True, directed=directed)
    A.graph_attr.update(
        rankdir="LR",
        splines="spline",
        nodesep="0.35",
        ranksep="0.70",
        pad="0.10",
        margin="0",
        bgcolor="white",
        outputorder="edgesfirst",
    )

    if show_title and title:
        A.graph_attr.update(
            label=title,
            labelloc="t",
            labeljust="c",
            fontname="Helvetica",
            fontsize="12",
        )

    A.node_attr.update(
        shape="box",
        style="rounded,filled",
        fontname="Helvetica",
        fontsize="10.5",
        fontcolor="#111827",
        color=NODE_BORDER,
        penwidth="0.9",
        margin="0.11,0.07",
    )

    for node in DOMAIN_ORDER:
        A.add_node(
            node,
            label=DISPLAY_LABELS[node],
            fillcolor=DOMAIN_FILL[node],
        )

    # Invisible rank anchors keep the same broad visual layout in every figure.
    anchors = []
    for idx, stage in enumerate(DOMAIN_VISUAL_STAGES):
        anchor = f"__rank_{idx}"
        anchors.append(anchor)
        A.add_node(anchor, shape="point", width="0", height="0", label="", style="invis")
        A.add_subgraph([anchor] + stage, name=f"rank_stage_{idx}", rank="same")

    for left, right in zip(anchors[:-1], anchors[1:]):
        A.add_edge(left, right, style="invis", weight="100", minlen="1")

    return A


# -----------------------------------------------------------------------------
# MATRIX VISUALISATIONS
# -----------------------------------------------------------------------------
def plot_precision_heatmap(
    precision,
    title=None,
    filename=None,
    figsize=(6.8, 5.5),
    annotate=True,
):
    """Plot a compact, thesis-ready precision-matrix heatmap."""
    values = np.asarray(precision.values, dtype=float)
    vmax = np.nanmax(np.abs(values))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="equal")

    labels = list(precision.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)

    if title:
        ax.set_title(title, fontsize=11, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Precision entry", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    if annotate:
        threshold = 0.60 * vmax if vmax > 0 else 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                text_color = "white" if abs(value) > threshold else "#111827"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    ax.tick_params(length=0)
    fig.tight_layout(pad=0.4)

    out = _normalise_output_path(filename)
    if out is not None:
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        if out.suffix.lower() != ".pdf":
            fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    plt.show()
    return fig, ax


def plot_partial_correlation_heatmap(
    partial_corr,
    title=None,
    filename=None,
    figsize=(6.8, 5.5),
    annotate=True,
):
    """Plot a partial-correlation heatmap, usually easier to interpret than precision values."""
    values = np.asarray(partial_corr.values, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")

    labels = list(partial_corr.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)

    if title:
        ax.set_title(title, fontsize=11, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Partial correlation", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7)

    ax.tick_params(length=0)
    fig.tight_layout(pad=0.4)

    out = _normalise_output_path(filename)
    if out is not None:
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        if out.suffix.lower() != ".pdf":
            fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    plt.show()
    return fig, ax


# -----------------------------------------------------------------------------
# DOMAIN-LEVEL CAUSAL GRAPHS
# -----------------------------------------------------------------------------
def visualize_causal_dag(
    edges,
    block_names,
    title=None,
    filename=None,
    show_title=False,
    show=True,
):
    """Render a clean directed ATTICA domain graph.

    Notes
    -----
    * The fixed stage layout is for visual consistency only.
    * Edge color is intentionally neutral; arrowheads carry direction.
    * The figure contains no "confounder/mediator" legend because those roles
      are exposure-outcome-query specific.
    """
    valid_nodes = set(block_names)
    A = _base_domain_graph(directed=True, title=title, show_title=show_title)

    for source, target in edges:
        if source not in valid_nodes or target not in valid_nodes:
            print(f"Skipped invalid edge: {source} -> {target}")
            continue
        A.add_edge(
            source,
            target,
            color=EDGE_COLOR,
            penwidth="1.05",
            arrowsize="0.65",
        )

    return _draw_graphviz(A, filename=filename, show=show, also_png=True)


def visualize_undirected_skeleton(
    edges,
    block_names,
    title=None,
    filename=None,
    show_title=False,
    show_weights=True,
    show=True,
):
    """Render a clean undirected sparse dependency skeleton.

    ``edges`` may contain ``(u, v)`` or ``(u, v, partial_correlation)``.
    When weights are supplied, line width represents absolute magnitude and
    negative partial correlations are dashed. By default, the graph also prints
    the numerical edge weights so the Lasso/Graphical-Lasso skeleton can be
    read directly from the figure.
    """
    valid_nodes = set(block_names)
    A = _base_domain_graph(directed=False, title=title, show_title=show_title)

    for edge in edges:
        if len(edge) == 3:
            source, target, weight = edge
            weight = float(weight)
        else:
            source, target = edge
            weight = None

        if source not in valid_nodes or target not in valid_nodes:
            print(f"Skipped invalid pair: {source} -- {target}")
            continue

        attrs = {
            "color": EDGE_COLOR,
            "penwidth": "1.0",
        }

        if weight is not None:
            attrs["penwidth"] = f"{0.70 + 4.2 * abs(weight):.2f}"
            attrs["style"] = "dashed" if weight < 0 else "solid"
            if show_weights:
                attrs.update(
                    label=f"{weight:.2f}",
                    fontname="Helvetica",
                    fontsize="8",
                    fontcolor="#334155",
                )

        A.add_edge(source, target, **attrs)

    return _draw_graphviz(A, filename=filename, show=show, also_png=True)


# -----------------------------------------------------------------------------
# VARIABLE-LEVEL / BIOMARKER GRAPH
# -----------------------------------------------------------------------------
def visualize_hierarchical_variable_dag(
    edges,
    upstream_vars,
    middle_vars,
    downstream_vars,
    title=None,
    filename=None,
    show_title=False,
    show=True,
):
    """Render a variable-level DAG as a compact three-stage left-to-right graph."""
    _require_graphviz()

    all_vars = list(dict.fromkeys(upstream_vars + middle_vars + downstream_vars))
    valid = set(all_vars)

    A = pgv.AGraph(strict=True, directed=True)
    A.graph_attr.update(
        rankdir="LR",
        splines="spline",
        nodesep="0.28",
        ranksep="0.85",
        pad="0.10",
        margin="0",
        bgcolor="white",
        outputorder="edgesfirst",
    )

    if show_title and title:
        A.graph_attr.update(
            label=title,
            labelloc="t",
            fontname="Helvetica",
            fontsize="12",
        )

    A.node_attr.update(
        shape="box",
        style="rounded,filled",
        fontname="Helvetica",
        fontsize="9.5",
        color=NODE_BORDER,
        penwidth="0.85",
        margin="0.10,0.06",
    )

    for node in upstream_vars:
        A.add_node(node, fillcolor="#F4F1EA")
    for node in middle_vars:
        A.add_node(node, fillcolor="#EDF2F7")
    for node in downstream_vars:
        A.add_node(node, fillcolor="#F5EFE7")

    stages = [upstream_vars, middle_vars, downstream_vars]
    anchors = []
    for idx, stage in enumerate(stages):
        anchor = f"__var_rank_{idx}"
        anchors.append(anchor)
        A.add_node(anchor, shape="point", width="0", height="0", label="", style="invis")
        A.add_subgraph([anchor] + list(stage), name=f"var_stage_{idx}", rank="same")

    for left, right in zip(anchors[:-1], anchors[1:]):
        A.add_edge(left, right, style="invis", weight="100")

    for source, target in edges:
        if source not in valid or target not in valid:
            print(f"Skipped invalid edge: {source} -> {target}")
            continue
        A.add_edge(source, target, color=EDGE_COLOR, penwidth="0.95", arrowsize="0.60")

    return _draw_graphviz(A, filename=filename, show=show, also_png=True)
