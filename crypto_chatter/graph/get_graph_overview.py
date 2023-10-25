import networkx as nx
import time
import numpy as np
import json

from crypto_chatter.data import load_graph_components
from .crypto_graph import CryptoGraph

def get_graph_overview(
    graph: CryptoGraph,
    recompute: bool = False,
) -> dict[str,any]:
    if not graph.data_config.graph_stats_file.is_file() or recompute:
        graph.load_components()
        start = time.time()
        longest_path = nx.dag_longest_path(graph.G)
        print(f'found longest path in {int(time.time() - start)} seconds')

        start = time.time()
        _, in_degree = zip(*graph.G.in_degree(graph.nodes))
        in_degree = np.array(in_degree)
        print(f'computed in_degree stats in {int(time.time() - start)} seconds')

        start = time.time()
        _, out_degree = zip(*graph.G.out_degree(graph.nodes))
        out_degree = np.array(out_degree)
        print(f'computed out_degree stats in {int(time.time() - start)} seconds')

        components_size = np.array([len(cc) for cc in graph.components])
        top_5_components = [
            {
                'id': int(cid), 
                'size': int(components_size[cid])
            }
            for cid in components_size.argsort()[:5:-1]
        ]

        print(f'why do I have {len(top_5_components)} components...')

        graph_stats = {
            "Node Count": len(graph.nodes),
            "Edge Count": len(graph.edges),
            "Longest Path": len(longest_path),
            "Connected Components Count": len(graph.components),
            "In-Degree": {
                "Max": int(in_degree.max()),
                "Avg": float(in_degree.mean()),
                "Min": int(in_degree.min()),
            },
            "Out-Degree": {
                "Max": int(out_degree.max()),
                "Avg": float(out_degree.mean()),
                "Min": int(out_degree.min()),
            },
            "Conncted Compoenents Size": {
                "Max": int(components_size.max()),
                "Avg": int(components_size.mean()),
                "Min": int(components_size.min()),
                "Top 5 Components": top_5_components
            }
        }

        json.dump(graph_stats, open(graph.data_config.graph_stats_file, 'w'), indent=2)
        return graph_stats

    else:
        graph_stats = json.load(open(graph.data_config.graph_stats_file))
        return graph_stats
