import time
import networkx as nx
import numpy as np
import json
from collections import Counter
from pathlib import Path
from rich.progress import Progress

from crypto_chatter.config import CryptoChatterGraphConfig
from crypto_chatter.data import CryptoChatterData
from crypto_chatter.utils.types import (
    NodeList,
    EdgeList,
    EdgeAttributeKind,
    NodeAttributeKind,
    SubGraphKind,
    ReachableKind,
    CentralityKind,
    DegreeKind,
)

from .degree import compute_degree
from .centrality import compute_centrality
from .edge_attributes import get_edge_attribute
from .node_attributes import get_node_attribute
from .build_graph import build_graph
from .reachable import get_reachable

class CryptoChatterGraph:
    G: nx.DiGraph
    nodes: NodeList
    edges: EdgeList
    data: CryptoChatterData
    graph_config: CryptoChatterGraphConfig
    data_source: str
    top_n_components: int 
    components: list[NodeList] | None = None
    progress: Progress|None
    use_progress: bool = False

    def __init__(
        self, 
        data: CryptoChatterData,
        graph_config: CryptoChatterGraphConfig,
        progress: Progress|None = None,
    ) -> None:
        self.graph_config = graph_config
        self.data = data
        self.progress = progress
        self.use_progress = progress is not None
        self.build(data)

    def build(
        self,
        data: CryptoChatterData,
    ) -> None:
        """
        Build the graph using the data from snapshot
        """
        start = time.time()

        nodes, edges = build_graph(
            data=data,
            graph_config=self.graph_config,
        )
        G = nx.DiGraph(edges)

        self.G = G
        self.nodes = nodes
        self.edges = edges

        print(f"constructed complete reply graph in {time.time()-start:.2f} seconds")

    def degree(
        self,
        kind: DegreeKind,
    ) -> np.ndarray:
        return compute_degree(
            G=self.G, 
            nodes=self.nodes, 
            graph_config=self.graph_config,
            kind=kind
        )

    def centrality(
        self,
        kind: CentralityKind
    ) -> np.ndarray:
        return compute_centrality(
            G=self.G, 
            nodes=self.nodes, 
            graph_config=self.graph_config,
            kind=kind
        )

    def reachable(
        self,
        node:int,
        kind: ReachableKind,
    ) -> NodeList:
        return get_reachable(
            G=self.G,
            node=node,
            kind=kind,
        )

    def get_stats(
        self,
        recompute: bool = False,
        display: bool = False
    ) -> dict[str, any]:
        save_file = self.graph_config.graph_dir / "stats/overview.json"

        if not save_file.is_file() or recompute:
            reply_count = (~self.data["quoted_status.id"].isna()).sum()

            start = time.time()
            longest_path = nx.dag_longest_path(self.G)
            print(f"found longest path in {time.time() - start:.2f} seconds")

            in_degree = self.degree("in")
            out_degree = self.degree("out")
            deg_cent = self.centrality("degree")
            in_deg_cent = self.centrality("in_degree")
            # bet_cent = self.betweenness_centrality()
            eig_cent = self.centrality("eigenvector")
            cls_cent = self.centrality("closeness")
            
            # self.load_components()
            # components_size = np.array([len(cc) for cc in self.components])

            graph_stats = {
                "Reply Tweets": {
                    "Count": f"{reply_count:,}",
                    "Ratio": f"{reply_count/len(self.data)*100:.2f}%"
                },
                "Standalone Tweets": {
                    "Count": f"{len(self.data)-reply_count:,}",
                    "Ratio": f"{(len(self.data)-reply_count)/len(self.data)*100:.2f}%"
                },
                "Node Count": len(self.nodes),
                "Edge Count": len(self.edges),
                "In-Degree": {
                    "Max": int(in_degree.max()),
                    "Avg": float(in_degree.mean()),
                    "Min": int(in_degree.min()),
                },
                "Out-Degree": {
                    "Max": int(out_degree.max()),
                    "Avg": float(out_degree.mean()),
                    "Min": int(out_degree.min()),
                },
                "Longest Path": len(longest_path),
                # "Connected Components Count": len(self.components),
                # "Conncted Compoenents Size": {
                #     "Max": int(components_size.max()),
                #     "Avg": int(components_size.mean()),
                #     "Min": int(components_size.min()),
                # },
                "Degree Centrality": {
                    "Max": deg_cent.max(),
                    "Avg": deg_cent.mean(),
                    "Min": deg_cent.min(),
                },
                "InDegree Centrality": {
                    "Max": in_deg_cent.max(),
                    "Avg": in_deg_cent.mean(),
                    "Min": in_deg_cent.min(),
                },
                # "Betweenness Centrality": {
                #     "Max": bet_cent.max(),
                #     "Avg": bet_cent.mean(),
                #     "Min": bet_cent.min(),
                # },
                "Eigenvector Centrality": {
                    "Max": eig_cent.max(),
                    "Avg": eig_cent.mean(),
                    "Min": eig_cent.min(),
                },
                "Closeness Centrality": {
                    "Max": cls_cent.max(),
                    "Avg": cls_cent.mean(),
                    "Min": cls_cent.min(),
                },
            }
            json.dump(graph_stats, open(save_file, "w"), indent=2)
        else:
            graph_stats = json.load(open(save_file))

        if display:
            print(json.dumps(graph_stats, indent=2))

        return graph_stats
