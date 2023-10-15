import networkx as nx
import pandas as pd
import time
import json

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.data import load_graph_data

from .get_graph_overview import get_graph_overview

class CryptoGraph:
    graph: nx.DiGraph | None = None
    nodes: list[int] | None = None
    edges: list[list[int]] | None = None
    data: pd.DataFrame | None = None
    data_config: CryptoChatterDataConfig
    data_source: str

    def build(self) -> None:
        graph_data, nodes, edges = load_graph_data(self.data_config)
        G = nx.DiGraph(edges)

        start = time.time()
        self.graph = G
        self.nodes = nodes
        self.edges = edges
        self.data = graph_data

        print(f'loaded complete reply graph in {int(time.time()-start)} seconds')

    def get_stats(
        self,
        recompute: bool = False,
        display: bool = False,
    ) -> dict[str,any]:
        if not self.nodes or not self.edges:
            raise Exception('Graph needs to be built first!')
        stats = get_graph_overview(
            nodes = self.nodes, 
            edges = self.edges, 
            data_config=self.data_config,
            recompute=recompute
        )
        if display:
            print(json.dumps(stats, indent=2))
        return stats

