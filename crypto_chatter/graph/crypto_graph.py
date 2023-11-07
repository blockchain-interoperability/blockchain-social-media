from typing_extensions import Self
import networkx as nx
import pandas as pd

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
