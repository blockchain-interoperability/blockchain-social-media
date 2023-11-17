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
    graph_config:CryptoChatterGraphConfig,
    kind: CentralityKind,
) -> np.ndarray:
    if not graph_config.is_directed and kind in typing.get_args(DirectedCentralityKind):
        raise ValueError(f"Centrality kind [{kind}] is not supported for undirected graphs")
    if kind not in centrality_functions:
        raise ValueError(f"Centrality kind [{kind}] is not supported")

    save_file = graph_config.graph_dir / f"stats/centrality/{kind}.npy"
    save_file.parent.mkdir(exist_ok=True, parents=True)
    if not save_file.is_file():
        start = time.time()
        values = centrality_functions[kind](G)
        by_nodes = np.array([values[n] for n in nodes])
        np.save(open(save_file, 'wb'), by_nodes)
        print(f"Computed {kind} centrality in {int(time.time() - start)} seconds")
    else:
        by_nodes = np.load(open(save_file, 'rb'))

    return by_nodes
