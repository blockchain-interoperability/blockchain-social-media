from typing_extensions import Self
import networkx as nx
import numpy as np
import time
import json

from .crypto_graph import CryptoGraph
from .load_reply_graph_data import load_reply_graph_data
from .load_weakly_connected_components import load_weaky_connected_components 

class CryptoTwitterTweetGraph(CryptoGraph):
    data_source = 'twitter'
    def build(self) -> None:
        '''
        Build the graph using the data from snapshot
        '''
        start = time.time()

        graph_data, nodes, edges = load_reply_graph_data(self.data_config)
        G = nx.DiGraph(edges)

        self.G = G
        self.nodes = nodes
        self.edges = edges
        self.data = graph_data

        print(f'constructed complete reply graph in {int(time.time()-start)} seconds')

    def load_components(
        self,
    ) -> Self:
        if self.components is None:
            self.components = load_weaky_connected_components(self)
        return self
    
    def in_degree(
        self,
    ) -> np.ndarray:
        save_file = self.data_config.graph_stats_dir / 'in_degree.json'
        if not save_file.is_file():
            start = time.time()
            in_degree = list(dict(self.G.in_degree(self.nodes)).values())
            print(f'computed in_degree stats in {int(time.time() - start)} seconds')
            json.dump(in_degree, open(save_file, 'w'))
        else:
            in_degree = json.load(open(save_file))
        return np.array(in_degree)

    def out_degree(
        self,
    ) -> np.ndarray:
        save_file = self.data_config.graph_stats_dir / 'out_degree.json'
        if not save_file.is_file():
            start = time.time()
            out_degree = list(dict(self.G.out_degree(self.nodes)).values())
            print(f'computed out_degree stats in {int(time.time() - start)} seconds')
            json.dump(out_degree, open(save_file, 'w'))
        else:
            out_degree = json.load(open(save_file))
        return np.array(out_degree)

    def get_stats(
        self,
        recompute: bool = False,
        display: bool = False
    ) -> dict[str, any]:
        overview_file = self.data_config.graph_stats_dir / 'overview.json'
        if not overview_file.is_file() or recompute:
            reply_count = (~self.data['quoted_status.id'].isna()).sum()

            start = time.time()
            longest_path = nx.dag_longest_path(self.G)
            print(f'found longest path in {int(time.time() - start)} seconds')

            in_degree = self.in_degree()
            out_degree = self.out_degree()
            deg_cent = self.degree_centrality()
            # bet_cent = self.betweenness_centrality()
            eig_cent = self.eigenvector_centrality()
            cls_cent = self.closeness_centrality()
            
            # self.load_components()
            # components_size = np.array([len(cc) for cc in self.components])

            graph_stats = {
                "Reply Tweets": {
                    "Count": f"{reply_count:,}",
                    "Ratio": f"{reply_count/len(self.data)*100:.2f}%"
                },
                "Standalone Tweets": {
                    "Count": f"{len(self.data)-reply_count:,}",
                    "Ratio": f"{(len(self.data)-reply_count)/len(self.data)*100:.2f}%"
                },
                "Node Count": len(self.nodes),
                "Edge Count": len(self.edges),
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
                # "Connected Components Count": len(self.components),
                # "Conncted Compoenents Size": {
                #     "Max": int(components_size.max()),
                #     "Avg": int(components_size.mean()),
                #     "Min": int(components_size.min()),
                # },
                "Degree Centrality": {
                    "Max": int(deg_cent.max()),
                    "Avg": int(deg_cent.mean()),
                    "Min": int(deg_cent.min()),
                },
                # "Betweenness Centrality": {
                #     "Max": int(bet_cent.max()),
                #     "Avg": int(bet_cent.mean()),
                #     "Min": int(bet_cent.min()),
                # },
                "Eigenvector Centrality": {
                    "Max": int(eig_cent.max()),
                    "Avg": int(eig_cent.mean()),
                    "Min": int(eig_cent.min()),
                },
                "Closeness Centrality": {
                    "Max": int(cls_cent.max()),
                    "Avg": int(cls_cent.mean()),
                    "Min": int(cls_cent.min()),
                },
            }
            json.dump(graph_stats, open(overview_file, 'w'), indent=2)
        else:
            graph_stats = json.load(open(overview_file))

        if display:
            print(json.dumps(graph_stats, indent=2))

        return graph_stats
