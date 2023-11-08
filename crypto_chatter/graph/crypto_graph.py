from typing_extensions import Self
import time
import networkx as nx
import numpy as np
import pandas as pd
import json

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import NodeList, EdgeList

class CryptoGraph:
    G: nx.DiGraph
    nodes: NodeList
    edges: EdgeList
    data: pd.DataFrame
    data_config: CryptoChatterDataConfig
    node_id_col: str
    data_source: str
    top_n_components: int 
    components: list[NodeList] | None = None

    def __init__(self, data_config: CryptoChatterDataConfig) -> None:
        self.data_config = data_config
        self.build()

    def build(self) -> None:
        ...

    def load_components(
        self,
    ) -> Self:
        ...

    def degree_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.data_config.graph_stats_dir / 'degree_centrality.json'
        if not save_file.is_file():
            start = time.time()
            deg_cent = nx.degree_centrality(self.G)
            deg_cent_values = [deg_cent[n] for n in self.nodes]
            print(f'computed degree centrality in {int(time.time() - start)} seconds')
        else:
            deg_cent_values = json.load(open(save_file))
        return np.array(deg_cent_values)

    def betweenness_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.data_config.graph_stats_dir / 'betweenness_centrality.json'
        if not save_file.is_file():
            start = time.time()
            bet_cent = nx.betweenness_centrality(self.G)
            bet_cent_values = [bet_cent[n] for n in self.nodes]
            print(f'computed betweenness centrality in {int(time.time() - start)} seconds')
        else:
            bet_cent_values = json.load(open(save_file))
        return np.array(bet_cent_values)

    def eigenvector_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.data_config.graph_stats_dir / 'eigenvector_centrality.json'
        if not save_file.is_file():
            start = time.time()
            eig_cent = nx.eigenvector_centrality(self.G)
            eig_cent_values = [eig_cent[n] for n in self.nodes]
            print(f'computed eigenvector centrality in {int(time.time() - start)} seconds')
        else:
            eig_cent_values = json.load(open(save_file))
        return np.array(eig_cent_values)

    def closeness_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.data_config.graph_stats_dir / 'closeness_centrality.json'
        if not save_file.is_file():
            start = time.time()
            eig_cent = nx.closeness_centrality(self.G)
            eig_cent_values = [eig_cent[n] for n in self.nodes]
            print(f'computed closeness centrality in {int(time.time() - start)} seconds')
        else:
            eig_cent_values = json.load(open(save_file))
        return np.array(eig_cent_values)

    def get_stats(
        self,
        recompute: bool = False,
        display: bool = False,
    ) -> dict[str, any]:
        overview_file = self.data_config.graph_stats_dir / 'overview.json'
        if not overview_file.is_file() or recompute:
            start = time.time()
            longest_path = nx.dag_longest_path(self.G)
            print(f'found longest path in {int(time.time() - start)} seconds')

            start = time.time()
            _, in_degree = zip(*self.G.in_degree(self.nodes))
            in_degree = np.array(in_degree)
            print(f'computed in_degree stats in {int(time.time() - start)} seconds')

            start = time.time()
            _, out_degree = zip(*self.G.out_degree(self.nodes))
            out_degree = np.array(out_degree)
            print(f'computed out_degree stats in {int(time.time() - start)} seconds')

            deg_cent = self.degree_centrality()
            bet_cent = self.betweenness_centrality()
            eig_cent = self.eigenvector_centrality()
            
            # self.load_components()
            # components_size = np.array([len(cc) for cc in self.components])
            
            reply_count = (~self.data['quoted_status.id'].isna()).sum()

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
        else:
            graph_stats = json.load(open(overview_file))

        if display:
            print(json.dumps(graph_stats, indent=2))

        return graph_stats

    def export_gephi_components(
        self,
    ) -> None:
        ...
