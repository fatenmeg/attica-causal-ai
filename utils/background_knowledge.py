"""
background_knowledge.py

Background-knowledge utilities for domain-level causal discovery.

The expert cardiovascular DAG is the single source of prior directional
knowledge. It combines biomedical plausibility with the baseline-to-follow-up
causal ordering. No separate ordering structure is used.
"""

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode


DEFAULT_BLOCKS = [
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


# Expert cardiovascular DAG used as the prior causal structure.
# These edges encode both biomedical plausibility and causal/temporal ordering.
# They are directional assumptions, not data-derived causal ground truth.
EXPERT_DOMAIN_EDGES = [
    # Upstream baseline determinants
    ("Demographics", "SES"),
    ("Demographics", "Behavioral"),
    ("Demographics", "Treatment"),
    ("Demographics", "Obesity"),
    ("Demographics", "Metabolic"),
    ("Demographics", "Conditions"),
    ("Demographics", "Outcomes"),

    ("SES", "Behavioral"),
    ("SES", "Treatment"),
    ("SES", "Obesity"),
    ("SES", "Metabolic"),
    ("SES", "Outcomes"),

    ("Behavioral", "Treatment"),
    ("Behavioral", "Obesity"),
    ("Behavioral", "Metabolic"),
    ("Behavioral", "Conditions"),
    ("Behavioral", "Outcomes"),

    # Mediterranean-diet exposure
    ("Treatment", "Obesity"),
    ("Treatment", "Metabolic"),
    ("Treatment", "Biomarkers"),
    ("Treatment", "Conditions"),
    ("Treatment", "Outcomes"),

    # Cardiometabolic pathways
    ("Obesity", "Metabolic"),
    ("Obesity", "Biomarkers"),
    ("Obesity", "Conditions"),
    ("Obesity", "Outcomes"),

    ("Metabolic", "Biomarkers"),
    ("Metabolic", "Conditions"),
    ("Metabolic", "Outcomes"),

    ("Biomarkers", "Conditions"),
    ("Biomarkers", "Outcomes"),

    # Final clinical pathway
    ("Conditions", "Outcomes"),
]


def canonical_pair(node_a, node_b):
    """Represent an undirected node pair consistently."""
    return tuple(sorted((node_a, node_b)))


def _validate_expert_edges(blocks_to_use, expert_edges):
    """Validate node names, duplicates, self-loops, and acyclicity."""
    block_set = set(blocks_to_use)

    unknown = [
        edge
        for edge in expert_edges
        if edge[0] not in block_set or edge[1] not in block_set
    ]
    if unknown:
        raise ValueError(
            "Expert edges contain unknown domains: "
            f"{unknown}"
        )

    self_loops = [edge for edge in expert_edges if edge[0] == edge[1]]
    if self_loops:
        raise ValueError(f"Expert DAG contains self-loops: {self_loops}")

    if len(expert_edges) != len(set(expert_edges)):
        raise ValueError("Expert DAG contains duplicate directed edges.")

    edge_set = set(expert_edges)
    reversed_pairs = [
        (source, target)
        for source, target in expert_edges
        if (target, source) in edge_set
    ]
    if reversed_pairs:
        raise ValueError(
            "Expert DAG contains edges in both directions for a pair: "
            f"{reversed_pairs}"
        )

    # Lightweight cycle check to keep this utility independent of NetworkX.
    children = {name: [] for name in blocks_to_use}
    indegree = {name: 0 for name in blocks_to_use}
    for source, target in expert_edges:
        children[source].append(target)
        indegree[target] += 1

    queue = [name for name in blocks_to_use if indegree[name] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if visited != len(blocks_to_use):
        raise ValueError("Expert background structure must be acyclic.")


def _source_and_sink_nodes(blocks_to_use, expert_edges):
    """Infer source-like and sink-like nodes from the expert DAG."""
    indegree = {name: 0 for name in blocks_to_use}
    outdegree = {name: 0 for name in blocks_to_use}

    for source, target in expert_edges:
        outdegree[source] += 1
        indegree[target] += 1

    source_nodes = {
        name for name in blocks_to_use if indegree[name] == 0
    }
    sink_nodes = {
        name for name in blocks_to_use if outdegree[name] == 0
    }

    return source_nodes, sink_nodes


def _add_expert_direction_constraints(
    bk,
    nodes,
    blocks_to_use,
    expert_edges,
    enforce_source_sink=True,
):
    """
    Translate the expert DAG into causal-learn background knowledge.

    Expert adjacencies are not forced to exist. For each expert edge X -> Y,
    only the reverse direction Y -> X is forbidden. PC/FCI may still remove
    X-Y when the conditional-independence evidence supports its removal.

    Source/sink status inferred from the expert DAG is also respected. In the
    current domain graph this means no incoming edges to Demographics and no
    outgoing edges from Outcomes, without introducing a separate ordering layer.
    """
    for source, target in expert_edges:
        bk.add_forbidden_by_node(
            nodes[target],
            nodes[source],
        )

    source_nodes, sink_nodes = _source_and_sink_nodes(
        blocks_to_use,
        expert_edges,
    )

    if enforce_source_sink:
        for source_node in source_nodes:
            for other in blocks_to_use:
                if other != source_node:
                    bk.add_forbidden_by_node(
                        nodes[other],
                        nodes[source_node],
                    )

        for sink_node in sink_nodes:
            for other in blocks_to_use:
                if other != sink_node:
                    bk.add_forbidden_by_node(
                        nodes[sink_node],
                        nodes[other],
                    )

    return source_nodes, sink_nodes


def build_expert_background_knowledge(
    blocks_to_use=None,
    expert_edges=None,
    enforce_source_sink=True,
):
    """
    Build expert-derived background knowledge for PC/FCI.

    The expert DAG is the sole directional prior. It already combines
    biomedical and temporal/causal assumptions, so no separate ordering variable is used.

    Returns
    -------
    bk : BackgroundKnowledge
        causal-learn background-knowledge object.
    nodes : dict[str, GraphNode]
        GraphNode objects keyed by domain name.
    expert_edges : list[tuple[str, str]]
        Validated expert DAG edges used to construct the constraints.
    """
    if blocks_to_use is None:
        blocks_to_use = DEFAULT_BLOCKS.copy()
    else:
        blocks_to_use = list(blocks_to_use)

    if expert_edges is None:
        expert_edges = EXPERT_DOMAIN_EDGES.copy()
    else:
        expert_edges = list(expert_edges)

    _validate_expert_edges(blocks_to_use, expert_edges)

    nodes = {
        name: GraphNode(name)
        for name in blocks_to_use
    }
    bk = BackgroundKnowledge()

    _add_expert_direction_constraints(
        bk=bk,
        nodes=nodes,
        blocks_to_use=blocks_to_use,
        expert_edges=expert_edges,
        enforce_source_sink=enforce_source_sink,
    )

    return bk, nodes, expert_edges


def build_hybrid_background_knowledge(
    sparse_pairs,
    blocks_to_use=None,
    expert_edges=None,
    enforce_source_sink=True,
):
    """
    Build hybrid sparse-plus-expert background knowledge.

    The admissible undirected candidate skeleton is

        sparse Graphical-Lasso pairs UNION skeleton(expert DAG).

    Pairs outside this union are forbidden in both directions. For an
    expert-supported pair X -> Y, the reverse direction Y -> X is forbidden.
    Sparse-only pairs remain directionally open to PC/FCI except for source/sink
    restrictions implied by the expert DAG.
    """
    if blocks_to_use is None:
        blocks_to_use = DEFAULT_BLOCKS.copy()
    else:
        blocks_to_use = list(blocks_to_use)

    if expert_edges is None:
        expert_edges = EXPERT_DOMAIN_EDGES.copy()
    else:
        expert_edges = list(expert_edges)

    _validate_expert_edges(blocks_to_use, expert_edges)
    block_set = set(blocks_to_use)

    sparse_pairs = {
        canonical_pair(a, b)
        for a, b in sparse_pairs
        if a in block_set and b in block_set and a != b
    }

    expert_pairs = {
        canonical_pair(a, b)
        for a, b in expert_edges
    }

    allowed_pairs = sparse_pairs.union(expert_pairs)

    nodes = {
        name: GraphNode(name)
        for name in blocks_to_use
    }
    bk = BackgroundKnowledge()

    # Restrict adjacency eligibility to the hybrid candidate set.
    for source in blocks_to_use:
        for target in blocks_to_use:
            if source == target:
                continue

            if canonical_pair(source, target) not in allowed_pairs:
                bk.add_forbidden_by_node(
                    nodes[source],
                    nodes[target],
                )

    source_nodes, sink_nodes = _add_expert_direction_constraints(
        bk=bk,
        nodes=nodes,
        blocks_to_use=blocks_to_use,
        expert_edges=expert_edges,
        enforce_source_sink=enforce_source_sink,
    )

    return {
        "background_knowledge": bk,
        "nodes": nodes,
        "sparse_pairs": sparse_pairs,
        "expert_pairs": expert_pairs,
        "allowed_pairs": allowed_pairs,
        "expert_edges": expert_edges,
        "source_nodes": source_nodes,
        "sink_nodes": sink_nodes,
    }


def print_expert_dag(expert_edges):
    """Print the directed expert DAG used as background knowledge."""
    print("=" * 80)
    print("EXPERT CARDIOVASCULAR DAG / BACKGROUND KNOWLEDGE")
    print("=" * 80)

    for source, target in expert_edges:
        print(f"{source} -> {target}")
