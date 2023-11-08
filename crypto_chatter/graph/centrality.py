import networkx as nx
import time
import numpy as np
import json

from .crypto_graph import CryptoGraph

def compute_degree_centrality(
    graph: CryptoGraph
) -> np.ndarray:
    save_file = graph.data_config.graph_stats_dir / 'degree_centrality.json'
    if not save_file.is_file():
        start = time.time()
        deg_cent = nx.degree_centrality(graph.G)
        deg_cent_values = [deg_cent[n] for n in graph.nodes]
        print(f'computed degree centrality in {int(time.time() - start)} seconds')
    else:
        deg_cent_values = json.load(open(save_file))
    return np.array(deg_cent_values)

def compute_betweenness_centrality(
    graph: CryptoGraph
) -> np.ndarray:
    save_file = graph.data_config.graph_stats_dir / 'betweenness_centrality.json'
    if not save_file.is_file():
        start = time.time()
        bet_cent = nx.betweenness_centrality(graph.G)
        bet_cent_values = [bet_cent[n] for n in graph.nodes]
        print(f'computed betweenness centrality in {int(time.time() - start)} seconds')
    else:
        bet_cent_values = json.load(open(save_file))
    return np.array(bet_cent_values)

def compute_eigenvector_centrality(
    graph: CryptoGraph
) -> np.ndarray:
    save_file = graph.data_config.graph_stats_dir / 'eigenvector_centrality.json'
    if not save_file.is_file():
        start = time.time()
        eig_cent = nx.eigenvector_centrality(graph.G)
        eig_cent_values = [eig_cent[n] for n in graph.nodes]
        print(f'computed eigenvector centrality in {int(time.time() - start)} seconds')
    else:
        eig_cent_values = json.load(open(save_file))
    return np.array(eig_cent_values)
