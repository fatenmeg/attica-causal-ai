"""
graph_utils.py

Utility functions for causal graph manipulation and validation.
"""

import networkx as nx
import pandas as pd


# =============================================================================
# EDGE EXTRACTION
# =============================================================================

def extract_directed_edges(graph, node_names):
    """
    Extract directed edges from a causal-learn graph.
    Directed edge pattern: i -> j is represented as G[i,j] = -1 and G[j,i] = 1.
    """
    if hasattr(graph, "graph"):
        G = graph.graph
    else:
        G = graph

    edges = []

    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if G[i, j] == -1 and G[j, i] == 1:
                edges.append((node_names[i], node_names[j]))

    return edges


def extract_undirected_edges(graph, node_names):
    """
    Extract undirected edges from a causal-learn graph.
    """
    if hasattr(graph, "graph"):
        G = graph.graph
    else:
        G = graph

    edges = []

    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):
            if G[i, j] == -1 and G[j, i] == -1:
                edges.append((node_names[i], node_names[j]))

    return edges


def extract_bidirected_edges(graph, node_names):
    """
    Extract bidirected edges from a causal-learn graph.
    """
    if hasattr(graph, "graph"):
        G = graph.graph
    else:
        G = graph

    edges = []

    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):
            if G[i, j] == 1 and G[j, i] == 1:
                edges.append((node_names[i], node_names[j]))

    return edges

# =============================================================================
# ENDPOINT CONSTRAINT VALIDATION
# =============================================================================

def check_endpoint_violations(
    edges,
    source_node="Demographics",
    sink_node="Outcomes",
):
    """
    Check that:
        - No edges point into the source node.
        - No edges originate from the sink node.

    Parameters
    ----------
    edges : list of tuples
        Directed edge list [(source, target), ...]

    source_node : str
        Node that should have no incoming edges.

    sink_node : str
        Node that should have no outgoing edges.
    """

    violations = []

    for source, target in edges:

        if target == source_node:
            violations.append(
                (source, target,
                 f"Incoming edge to {source_node}")
            )

        if source == sink_node:
            violations.append(
                (source, target,
                 f"Outgoing edge from {sink_node}")
            )

    print("\n" + "=" * 80)
    print("ENDPOINT CONSTRAINT CHECK")
    print("=" * 80)

    if violations:

        print(f"⚠ {len(violations)} endpoint violations found:\n")

        for source, target, reason in violations:
            print(f"{source} → {target}   ({reason})")

    else:
        print("✓ No endpoint violations detected.")

    return violations
# =============================================================================
# GRAPH CONVERSION
# =============================================================================

def edge_list_to_digraph(edges, node_names=None):
    """
    Convert edge list to NetworkX directed graph.
    """
    G = nx.DiGraph()

    if node_names is not None:
        G.add_nodes_from(node_names)

    G.add_edges_from(edges)

    return G


def graph_to_edge_list(G):
    """
    Convert NetworkX graph to edge list.
    """
    return list(G.edges())


def remove_duplicate_edges(edges):
    """
    Remove duplicate directed edges while preserving order.
    """
    return list(dict.fromkeys(edges))


# =============================================================================
# GRAPH VALIDATION
# =============================================================================

def check_cycles(edges, graph_name="Graph", node_names=None):
    """
    Check whether a directed graph contains cycles.
    """
    G = edge_list_to_digraph(edges, node_names=node_names)

    print("\n" + "=" * 80)
    print(f"CYCLE CHECK: {graph_name}")
    print("=" * 80)

    if nx.is_directed_acyclic_graph(G):
        print("✓ No cycles detected. Graph is a DAG.")
    else:
        cycles = list(nx.simple_cycles(G))
        print(f"✗ Cycles detected: {len(cycles)}")
        for cycle in cycles:
            print("  ", cycle)

    return G


def is_acyclic(edges, node_names=None):
    """
    Return True if graph is a DAG.
    """
    G = edge_list_to_digraph(edges, node_names=node_names)
    return nx.is_directed_acyclic_graph(G)


def topological_order(edges, node_names=None):
    """
    Return topological ordering of a DAG.
    """
    G = edge_list_to_digraph(edges, node_names=node_names)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph contains cycles. Topological order is not defined.")

    return list(nx.topological_sort(G))


# =============================================================================
# GRAPH SIMPLIFICATION
# =============================================================================

def transitive_reduce_edges(edges, graph_name="Graph", node_names=None):
    """
    Apply transitive reduction to a DAG.
    If the graph has cycles, the original edges are returned.
    """
    G = edge_list_to_digraph(edges, node_names=node_names)

    print("\n" + "=" * 80)
    print(f"TRANSITIVE REDUCTION: {graph_name}")
    print("=" * 80)

    if not nx.is_directed_acyclic_graph(G):
        print("✗ Graph contains cycles. Skipping transitive reduction.")
        return edges

    G_reduced = nx.transitive_reduction(G)
    reduced_edges = list(G_reduced.edges())

    print(f"Original edges: {len(edges)}")
    print(f"Reduced edges : {len(reduced_edges)}")
    print(f"Removed edges : {len(edges) - len(reduced_edges)}")

    return reduced_edges


# =============================================================================
# GRAPH SUMMARY
# =============================================================================

def graph_statistics(edges, graph_name="Graph", node_names=None):
    """
    Return basic graph statistics.
    """
    G = edge_list_to_digraph(edges, node_names=node_names)

    stats = {
        "Graph": graph_name,
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": nx.density(G),
        "Is_DAG": nx.is_directed_acyclic_graph(G),
    }

    return pd.Series(stats)


def print_edges(edges, title="Edges"):
    """
    Pretty-print edge list.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not edges:
        print("No edges found.")
        return

    for source, target in edges:
        print(f"{source:20s} → {target}")


# =============================================================================
# EXPORT
# =============================================================================

def save_edges(edges, filename):
    """
    Save edge list to CSV.
    """
    pd.DataFrame(
        edges,
        columns=["Source", "Target"]
    ).to_csv(filename, index=False)