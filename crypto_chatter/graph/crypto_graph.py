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

    def degree(
        self,
    ):
        start = time.time()
        _, degree = zip(*self.G.degree(self.nodes))
        degree = np.array(degree)
        print(f'computed degree stats in {int(time.time() - start)} seconds')
        return degree

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
        ...

    def export_gephi_components(
        self,
    ) -> None:
        ...
