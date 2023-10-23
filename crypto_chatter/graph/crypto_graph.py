import networkx as nx
import pandas as pd

from crypto_chatter.config import CryptoChatterDataConfig

class CryptoGraph():
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
        ...

    def check_graph_is_built(self) -> bool:
        ...
