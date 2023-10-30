import networkx as nx
import time
import numpy as np
import json
from pathlib import Path

from .crypto_graph import CryptoGraph
from crypto_chatter.utils import EdgeList, NodeList

def get_graph_overview(
    G: nx.Graph,
    nodes: NodeList,
    edges: EdgeList,
    components: list[NodeList],
    graph_stats_file: Path,
    recompute: bool = False,
) -> dict[str,any]:
    if not graph_stats_file.is_file() or recompute:
        start = time.time()
        longest_path = nx.dag_longest_path(G)
        print(f'found longest path in {int(time.time() - start)} seconds')

        if isinstance(G, nx.DiGraph):
            start = time.time()
            _, in_degree = zip(*G.in_degree(nodes))
            in_degree = np.array(in_degree)
            print(f'computed in_degree stats in {int(time.time() - start)} seconds')

            start = time.time()
            _, out_degree = zip(*G.out_degree(nodes))
            out_degree = np.array(out_degree)
            print(f'computed out_degree stats in {int(time.time() - start)} seconds')
            degree_stats = {
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
            }
        else:
            _, degree = zip(*G.degree(nodes))
            degree_stats = {
                "Degree": {
                    "Max": int(degree.max()),
                    "Avg": float(degree.mean()),
                    "Min": int(degree.min()),
                }
            }

        components_size = np.array([len(cc) for cc in components])

        print(f'why do I have {len(top_5_components)} components from {len(components_size)}...')

        graph_stats = {
            "Node Count": len(nodes),
            "Edge Count": len(edges),
            "Longest Path": len(longest_path),
            **degree_stats,
            "Connected Components Count": len(components),
            "Conncted Compoenents Size": {
                "Max": int(components_size.max()),
                "Avg": int(components_size.mean()),
                "Min": int(components_size.min()),
            }
        }

        json.dump(graph_stats, open(graph_stats_file, 'w'), indent=2)
        return graph_stats

    else:
        graph_stats = json.load(open(graph_stats_file))
        return graph_stats
