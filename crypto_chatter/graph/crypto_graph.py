import networkx as nx
import pandas as pd
import json

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import NodeList, EdgeList
from crypto_chatter.utils import progress_bar

from .load_weakly_connected_components import load_weaky_connected_components
from .get_graph_overview import get_graph_overview

class CryptoGraph():
    G: nx.DiGraph
    nodes: NodeList
    edges: EdgeList
    data: pd.DataFrame
    data_config: CryptoChatterDataConfig
    node_id_col: str
    data_source: str
    top_n_components: int 
    components: NodeList | None = None

    def __init__(self, data_config: CryptoChatterDataConfig) -> None:
        self.data_config = data_config
        self.build()

    def build(self) -> None:
        ...

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
            self.components = load_weaky_connected_components(self.G, self.data_config)

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
                            self.data[self.data_config.node_id_col].values.tolist(), 
                            self.data[col].values.tolist()
                        )),
                        name = col,
                    )
                nx.write_gexf(subgraph, self.data_config.graph_gephi_dir / f'{i:06d}.gexf')
                progress.advance(save_task)

    def populate_attributes(self) -> None:
        ...

