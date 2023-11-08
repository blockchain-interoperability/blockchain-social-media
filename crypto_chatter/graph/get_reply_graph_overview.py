import networkx as nx
import time
import numpy as np
import json

from .crypto_graph import CryptoGraph

def get_reply_graph_overview(
    graph: CryptoGraph,
    recompute: bool = False,
) -> dict[str,any]:
    overview_file = graph.data_config.graph_stats_dir / 'overview.json'
    if not overview_file.is_file() or recompute:
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

        deg_cent = graph.degree_centrality()
        bet_cent = graph.betweenness_centrality()
        eig_cent = graph.eigenvector_centrality()
        
        reply_count = (~graph.data['quoted_status.id'].isna()).sum()
        components_size = np.array([len(cc) for cc in graph.components])

        graph_stats = {
            "Reply Tweets": {
                "Count": f"{reply_count:,}",
                "Ratio": f"{reply_count/len(graph.data)*100:.2f}%"
            },
            "Standalone Tweets": {
                "Count": f"{len(graph.data)-reply_count:,}",
                "Ratio": f"{(len(graph.data)-reply_count)/len(graph.data)*100:.2f}%"
            },
            "Node Count": len(graph.nodes),
            "Edge Count": len(graph.edges),
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
            "Longest Path": len(longest_path),
            "Connected Components Count": len(graph.components),
            "Conncted Compoenents Size": {
                "Max": int(components_size.max()),
                "Avg": int(components_size.mean()),
                "Min": int(components_size.min()),
            },
            "Degree Centrality": {
                "Max": int(deg_cent.max()),
                "Avg": int(deg_cent.mean()),
                "Min": int(deg_cent.min()),
            },
            "Betweenness Centrality": {
                "Max": int(bet_cent.max()),
                "Avg": int(bet_cent.mean()),
                "Min": int(bet_cent.min()),
            },
            "Eigenvector Centrality": {
                "Max": int(eig_cent.max()),
                "Avg": int(eig_cent.mean()),
                "Min": int(eig_cent.min()),
            },
        }
        json.dump(graph_stats, open(overview_file, 'w'), indent=2)
        return graph_stats

    else:
        graph_stats = json.load(open(overview_file))
        return graph_stats
