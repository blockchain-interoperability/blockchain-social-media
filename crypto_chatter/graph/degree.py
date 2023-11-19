import networkx as nx

import typing
import time
import networkx as nx
import numpy as np

from crypto_chatter.config import CryptoChatterGraphConfig
from crypto_chatter.utils.types import (
    NodeList,
    DirectedDegreeKind,
    DegreeKind,
)

def get_degree_func(
    G: nx.Graph,
    degree_kind: DegreeKind
):
    if degree_kind == 'all':
        return G.degree
    elif degree_kind == 'in':
        if not isinstance(G, nx.DiGraph):
            raise ValueError(f'graph must be directed to compute in_degree')
        return G.in_degree
    elif degree_kind == 'out':
        if not isinstance(G, nx.DiGraph):
            raise ValueError(f'graph must be directed to compute in_degree')
        return G.out_degree
    else:
        raise ValueError(f'invalid degree kind: [{degree_kind}]')

def compute_degree(
    G: nx.Graph,
    nodes: NodeList,
    graph_config:CryptoChatterGraphConfig,
    kind: DegreeKind,
) -> np.ndarray:
    if not graph_config.is_directed and kind in typing.get_args(DirectedDegreeKind):
        raise ValueError(f"Degree kind [{kind}] is not supported for undirected graphs")

    start = time.time()
    save_file = graph_config.graph_dir / f"stats/degree/{kind}.npy"
    save_file.parent.mkdir(exist_ok=True, parents=True)
    if not save_file.is_file():
        values = get_degree_func(G, kind)(nodes)
        by_nodes = np.array([values[n] for n in nodes])
        np.save(open(save_file, 'wb'), by_nodes)
    else:
        by_nodes = np.load(open(save_file, 'rb'))

    print(f"Computed {kind} centrality in {time.time() - start:.2f} seconds")
    return by_nodes

