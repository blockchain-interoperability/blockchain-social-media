import networkx as nx
import time

from crypto_chatter.data import (
    load_graph_data, 
    load_reply_graph_edges,
)

from .crypto_graph import CryptoGraph

class CryptoTwitterReplyGraph(CryptoGraph):
    data_source = 'twitter'

    def build(self) -> None:
        '''
        Build the graph using the data from snapshot
        '''
        start = time.time()

        nodes, edges = load_reply_graph_edges(self.data_config)
        graph_data = load_graph_data(nodes, self.data_config)
        G = nx.DiGraph(edges)

        self.G = G
        self.nodes = nodes
        self.edges = edges
        self.data = graph_data

        print(f'constructed complete reply graph in {int(time.time()-start)} seconds')

    def populate_attributes(self):
        for attr in self.data_config.graph_attributes:
            nx.set_node_attributes(
                self.G,
                self.data[attr],
                name = attr,
            )
