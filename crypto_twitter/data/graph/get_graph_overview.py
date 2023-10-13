import networkx as nx
import time
import numpy as np
import json

from crypto_twitter.config import GRAPH_STATS_FILE
from .load_graph_edges import load_graph_edges

def get_graph_overview(recompute: bool = False) -> None:
    if not GRAPH_STATS_FILE.is_file() or recompute:
        start = time.time()
        nodes,edges = load_graph_edges()
        G = nx.DiGraph(edges)

        start = time.time()
        longest_path = nx.dag_longest_path(G)
        print(f'found longest path in {int(time.time() - start)} seconds')

        start = time.time()
        _, in_degree = zip(*G.in_degree(nodes))
        in_degree = np.array(in_degree)
        print(f'computed in_degree  stats in {int(time.time() - start)} seconds')

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

        json.dump(graph_stats, open(GRAPH_STATS_FILE, 'w'))

    else:
        graph_stats = json.load(open(GRAPH_STATS_FILE))

        return graph_stats
