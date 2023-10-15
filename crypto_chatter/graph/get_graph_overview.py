import networkx as nx
import time
import numpy as np
import json

from crypto_chatter.config import CryptoChatterDataConfig

def get_graph_overview(
    nodes: list[int], 
    edges: list[list[int]],
    data_config: CryptoChatterDataConfig,
    recompute: bool = False
) -> dict[str,any]:
    if not data_config.graph_stats_file.is_file() or recompute:
        start = time.time()
        G = nx.DiGraph(edges)

        start = time.time()
        longest_path = nx.dag_longest_path(G)
        print(f'found longest path in {int(time.time() - start)} seconds')

        start = time.time()
        _, in_degree = zip(*G.in_degree(nodes))
        in_degree = np.array(in_degree)
        print(f'computed in_degree stats in {int(time.time() - start)} seconds')

        start = time.time()
        _, out_degree = zip(*G.out_degree(nodes))
        out_degree = np.array(out_degree)
        print(f'computed out_degree stats in {int(time.time() - start)} seconds')

        graph_stats = {
            "Node Count": len(nodes),
            "Edge Count": len(edges),
            "Longest Path": len(longest_path),
            "In-Degree": {
                "Max": int(in_degree.max()),
                "Avg": float(in_degree.mean()),
                "Min": int(in_degree.min()),
            },
            "Out-Degree": {
                "Max": int(out_degree.max()),
                "Avg": float(out_degree.mean()),
                "Min": int(out_degree.min()),
            }
        }

        json.dump(graph_stats, open(data_config.graph_stats_file, 'w'), indent=2)
        return graph_stats

    else:
        graph_stats = json.load(open(data_config.graph_stats_file))
        return graph_stats
