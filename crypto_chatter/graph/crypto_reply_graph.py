import networkx as nx
import pandas as pd
import time
import json

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.data import (
    load_graph_data, 
    load_graph_edges,
    load_graph_components,
)

from .get_graph_overview import get_graph_overview

class CryptoReplyGraph:
    G: nx.DiGraph | None = None
    nodes: list[int] | None = None
    edges: list[list[int]] | None = None
    components: list[list[int]] | None = None
    data: pd.DataFrame | None = None
    data_config: CryptoChatterDataConfig
    data_source: str

    def __init__(self, *args, **kwargs) -> None:
        ...

    def populate_attributes(self) -> None:
        ...

    def build(self) -> None:
        '''
        Build the graph using the data from snapshot
        '''
        start = time.time()

        nodes, edges = load_graph_edges(self.data_config)
        graph_data = load_graph_data(nodes, self.data_config)
        G = nx.DiGraph(edges)

        self.G = G
        self.nodes = nodes
        self.edges = edges
        self.data = graph_data
        self.components = load_graph_components(G, self.data_config)

        # self.populate_attributes()

        print(f'constructed complete reply graph in {int(time.time()-start)} seconds')

    def check_graph_is_built(self):
        '''
        Called before operations that require the actual graph. 
        Used to ensure that graph is built.
        '''
        if not self.nodes or not self.edges or not self.G:
            raise Exception('Graph needs to be built first!')

    def get_stats(
        self,
        recompute: bool = False,
        display: bool = False,
    ) -> dict[str,any]:
        '''
        Get basic statistics of the network. 
        '''
        self.check_graph_is_built()
        stats = get_graph_overview(
            G = self.G,
            data_config=self.data_config,
            recompute=recompute
        )
        if display:
            print(json.dumps(stats, indent=2))
        return stats
    
    def export_gephi_component(
        self,
        component_id: int,
    ) -> None:
        '''
        Export the selected component to a file format that can be consumed by gephi for visual inspection
        '''
        self.check_graph_is_built()
        self.components[component_id]
        ...


    def export_gephi_full(
        self,
    ) -> None:
        '''
        Export the full graph to a file format that can be consumed by gephi for visual inspection
        '''
        self.check_graph_is_built()
        full_graph_file = self.data_config.graph_gephi_dir / 'full.gexf'
        nx.write_gexf(self.G, full_graph_file)
        print(f'exported graph to {str(full_graph_file)}')

