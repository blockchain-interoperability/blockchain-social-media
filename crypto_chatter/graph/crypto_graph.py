import networkx as nx
import pandas as pd

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import NodeList, EdgeList

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

    def __init__(self, *args, **kwargs) -> None:
        ...

    def populate_attributes(self) -> None:
        ...

    def build(self) -> None:
        ...
