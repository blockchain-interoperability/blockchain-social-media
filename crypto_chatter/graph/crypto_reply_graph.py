import networkx as nx
import time
import json

from crypto_chatter.data import (
    load_graph_data, 
    load_reply_graph_edges,
    load_graph_components,
)
from crypto_chatter.utils import progress_bar

from .crypto_graph import CryptoGraph
from .get_graph_overview import get_graph_overview

class CryptoReplyGraph(CryptoGraph):
    def __init__(self, index_name:str, *args, **kwargs) -> None:
        self.configure(index_name=index_name)
        self.build()
        ...

    def configure(self, index_name: str) -> None:
        ...

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

    def get_stats(
        self,
        recompute: bool = False,
        display: bool = False,
    ) -> dict[str, any]:
        '''
        Get basic statistics of the network. 
        '''
        stats = get_graph_overview(graph=self, recompute=recompute)
        if display:
            print(json.dumps(stats, indent=2))
        return stats

    def load_components(
        self,
    ):
        if self.components is None:
            self.components = load_graph_components(graph=self)
    
    def export_gephi_components(
        self,
    ) -> None:
        self.load_components()
        with progress_bar() as progress:
            save_task = progress.add_task('exporting components to gephi..', total=len(self.components))
            for i,c in enumerate(self.components):
                subgraph = self.G.subgraph(c)
                for col in self.data.columns:
                    nx.set_node_attributes(
                        subgraph,
                        values = dict(zip(
                            self.data[self.data_config.node_id_col].values, 
                            self.data[col].values
                        )),
                        name = col,
                    )
                nx.write_gexf(subgraph, self.data_config.graph_gephi_dir / f'{i:06d}.gexf')
                progress.advance(save_task)

    def export_gephi_full(
        self,
    ) -> None:
        '''
        Export the full graph to a file format that can be consumed by gephi for visual inspection
        '''
        full_graph_file = self.data_config.graph_gephi_dir / 'full.gexf'
        nx.write_gexf(self.G, full_graph_file)
        print(f'exported graph to {str(full_graph_file)}')

