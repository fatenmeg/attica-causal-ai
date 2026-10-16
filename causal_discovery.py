"""
causal_discovery.py

Wrappers for PC and FCI causal discovery.
"""

import numpy as np
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

from utils.graph_utils import extract_directed_edges


def prepare_causal_data(df):
    """
    Convert dataframe to float numpy array for causal-learn.
    """
    df_clean = df.copy().dropna()

    data = df_clean.astype(np.float64).values
    node_names = list(df_clean.columns)

    return df_clean, data, node_names

def run_pc_discovery(
    data,
    node_names,
    background_knowledge=None,
    alpha=0.05,
    stable=True,
    show_progress=True,
):
    """
    Run PC algorithm and extract directed edges.
    """

    cg = pc(
        data,
        alpha=alpha,
        indep_test=fisherz,
        stable=stable,
        background_knowledge=background_knowledge,
        node_names=node_names,
        show_progress=show_progress,
    )

    directed_edges = extract_directed_edges(cg.G, node_names)

    return cg, directed_edges


def run_fci_discovery(
    data,
    node_names,
    background_knowledge=None,
    alpha=0.05,
    depth=-1,
    max_path_length=-1,
    show_progress=True,
):
    """
    Run FCI algorithm and extract directed edges.
    """

    result = fci(
        data,
        indep_test_method=fisherz,
        alpha=alpha,
        depth=depth,
        max_path_length=max_path_length,
        background_knowledge=background_knowledge,
        node_names=node_names,
        show_progress=show_progress,
    )

    graph = result[0] if isinstance(result, tuple) else result.G

    directed_edges = extract_directed_edges(graph, node_names)

    return graph, directed_edges