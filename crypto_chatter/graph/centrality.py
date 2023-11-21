import typing
import time
import networkx as nx
import numpy as np

from crypto_chatter.config import CryptoChatterGraphConfig
from crypto_chatter.utils.types import (
    NodeList,
    CentralityKind,
    DirectedCentralityKind,
)

centrality_functions = {
    'degree': nx.degree_centrality,
    'betweenness': nx.betweenness_centrality,
    'eigenvector': nx.eigenvector_centrality,
    'in_degree': nx.in_degree_centrality,
    'out_degree': nx.out_degree_centrality,
}

def compute_centrality(
    G: nx.Graph,
    nodes: NodeList,
    kind: CentralityKind,
) -> np.ndarray:
    if not isinstance(G, nx.DiGraph) and kind in typing.get_args(DirectedCentralityKind):
        raise ValueError(f"Centrality kind [{kind}] is not supported for undirected graphs")
    if kind not in centrality_functions:
        raise ValueError(f"Centrality kind [{kind}] is not supported")

    start = time.time()
    values = centrality_functions[kind](G)
    by_nodes = np.array([values[n] for n in nodes])
    print(f"Computed {kind} centrality in {time.time() - start:.2f} seconds")

    return by_nodes
