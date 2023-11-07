from typing_extensions import Self
import networkx as nx
import json
import time

from .crypto_graph import CryptoGraph
from .load_reply_graph_data import load_reply_graph_data
from .load_weakly_connected_components import load_weaky_connected_components 
from .get_reply_graph_overview import get_reply_graph_overview

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
    def get_stats(
        self,
        recompute: bool = False,
        display: bool = False,
    ) -> dict[str, any]:
        self.load_components()
        '''
        Get basic statistics of the network. 
        '''
        stats = get_reply_graph_overview(self, recompute=recompute)
        if display:
            print(json.dumps(stats, indent=2))
        return stats
