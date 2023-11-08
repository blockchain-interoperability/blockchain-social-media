import networkx as nx
import time
import numpy as np
import json

from .crypto_graph import CryptoGraph

def compute_degree_centrality(
    graph: CryptoGraph
) -> np.array:
    start = time.time()
    deg_cent = nx.degree_centrality(graph.G)
    deg_cent_values = np.array([deg_cent[n] for n in graph.nodes])
    print(f'computed degree centrality in {int(time.time() - start)} seconds')