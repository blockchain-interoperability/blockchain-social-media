import networkx as nx
import pandas as pd
import numpy as np
import json
import time

from crypto_twitter.data import (
    load_graph_data,
    load_graph_edges,
)
from .get_graph_overview import get_graph_overview

class TweetReplyGraph:
    graph: nx.DiGraph
    nodes: list[int]
    edges: list[list[int]]
    data: pd.DataFrame

    def __init__(self) -> None:
        nodes, edges = load_graph_edges()
        graph_data = load_graph_data()
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
        stats = get_graph_overview(recompute)
        if display:
            print(json.dumps(stats, indent=2))
        return stats

