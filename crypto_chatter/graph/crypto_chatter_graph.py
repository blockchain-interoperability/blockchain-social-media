from typing import Literal
from typing_extensions import Self
import time
import networkx as nx
import numpy as np
import pandas as pd
import json
from collections import Counter
from pathlib import Path

from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig
from crypto_chatter.data import CryptoChatterData
from crypto_chatter.utils.types import (
    NodeList,
    EdgeList,
    NodeToIndexMapping,
    EdgeAttributeKind,
    NodeAttributeKind,
    CentralityKind,
    DegreeKind,
)

# from .crypto_chatter_sub_graph import CryptoChatterSubGraph
from .degree import compute_degree
from .centrality import compute_centrality
from .edge_attributes import get_edge_attributes
from .node_attributes import get_node_attributes
from .load_graph_data import load_graph_data
# from .node_attributes import 

class CryptoChatterSubGraph:
    id: str
    parent: 'CryptoChatterGraph'
    # source: int
    nodes: NodeList
    edges: EdgeList
    graph: nx.Graph
    data: CryptoChatterData
    cache_dir: Path

    def __init__(
        self, 
        _id: str,
        parent: 'CryptoChatterGraph', 
        nodes: NodeList,
    ):
        self.id = _id
        self.parent = parent
        # self.source = source
        self.nodes = nodes
        self.graph = self.parent.G.subgraph(self.nodes)
        self.edges = self.graph.edges(self.nodes)
        self.cache_dir = self.parent.graph_config.graph_dir / f"subgraph/{_id}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data = CryptoChatterData(
            self.parent.data.data_config,
            df = self.parent.data.df.iloc[
                [
                    self.parent.node_to_index[n] 
                    for n in self.nodes 
                    if self.parent.node_to_index[n] != -1
                ]
            ]
        )

    def degree(
        self,
        kind: DegreeKind,
    ) -> np.ndarray:
        return self.parent.degree(kind)[self.nodes]

    def get_keywords(
        self,
        top_n: int = 100,
    ) -> dict[str, float]:
        save_file = self.cache_dir / f"keywords/{self.parent.data.tfidf_settings}/{top_n}.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        if not save_file.is_file():
            if self.parent.data.tfidf is None:
                self.parent.data.fit_tfidf()
            terms = self.parent.data.tfidf.get_feature_names_out()
            vecs = self.parent.data.tfidf.transform(self.data.text())
            tfidf_scores = vecs.toarray().sum(0)
            sorted_idxs = tfidf_scores.argsort()[::-1]
            keywords = terms[sorted_idxs][:top_n]
            keyword_scores = tfidf_scores[sorted_idxs][:top_n]
            keywords_with_score = dict(zip(keywords, keyword_scores))
            json.dump(keywords_with_score, open(save_file, "w"))
        else:
            keywords_with_score = json.load(open(save_file))

        return keywords_with_score

    def count_hashtags(
        self,
        top_n:int = 100,
    ) -> dict[str, int]:
        save_file = self.cache_dir / f"hashtags/{top_n}.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        if not save_file.is_file():
            hashtag_count = dict(
                Counter([
                    tag
                    for hashtags in self.data["hashtags"].values
                    for tag in hashtags
                ]).most_common()[:top_n]
            )
            json.dump(hashtag_count, open(save_file, "w"))
        else:
            hashtag_count = json.load(open(save_file))
        return hashtag_count
    
    def export_gephi(
        self,
        node_attributes: list[NodeAttributeKind] = [],
        edge_attributes: list[EdgeAttributeKind] = [],
    ) -> None:

        for att in node_attributes:
            nx.set_node_attributes()
            self.data[att]
        ...

class CryptoChatterGraph:
    G: nx.DiGraph
    nodes: NodeList
    edges: EdgeList
    node_to_index: NodeToIndexMapping
    data: CryptoChatterData
    graph_config: CryptoChatterGraphConfig
    data_source: str
    top_n_components: int 
    components: list[NodeList] | None = None

    def __init__(
        self, 
        graph_config: CryptoChatterGraphConfig,
        data_config: CryptoChatterDataConfig,
        columns: list[str]|None = None,
    ) -> None:
        self.graph_config = graph_config
        self.build(data_config, columns)

    def build(
        self,
        data_config:CryptoChatterDataConfig,
        columns: list[str]|None = None,
    ) -> None:
        """
        Build the graph using the data from snapshot
        """
        start = time.time()

        graph_data, nodes, edges, node_to_index = load_graph_data(
            data_config,
            self.graph_config,
            columns,
        )
        G = nx.DiGraph(edges)

        self.G = G
        self.nodes = nodes
        self.edges = edges
        self.node_to_index = node_to_index
        self.data = graph_data

        print(f"constructed complete reply graph in {int(time.time()-start)} seconds")

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

    def get_top_n_nodes(
        self,
        by_centrality: CentralityKind,
        top_n: int = 10,
    ) -> NodeList:
        selected = []
        centrality = self.centrality(by_centrality)
        centrality_idx = centrality.argsort()[::-1]
        counter = 0
        while len(selected) < top_n:
            cur_idx = centrality_idx[counter]
            if self.nodes[cur_idx] in self.data["id"].values:
                selected += [self.nodes[cur_idx]]
            counter += 1
        return selected

    def get_top_n_subgraphs(
        self,
        by_centrality: CentralityKind,
        top_n: int = 10,
    ) -> list[CryptoChatterSubGraph]:
        return [
            CryptoChatterSubGraph(
                f'{by_centrality}_{str(node)}',
                parent=self,
                nodes=self.get_all_reachable_nodes(node)
            )
            for node in self.get_top_n_nodes(by_centrality, top_n)
        ]

    def get_all_reachable_nodes(
        self, 
        node: int,
    ) -> NodeList:
        stack = [node]
        reachable = []
        while stack:
            current = stack.pop()
            reachable += [current]
            for neighbor in nx.all_neighbors(self.G, current):
                if neighbor not in reachable:
                    stack += [neighbor]
        return reachable

    def export_gephi(
        self,
    ) -> None:
        ...

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
            print(f"found longest path in {int(time.time() - start)} seconds")

            in_degree = self.degree('in')
            out_degree = self.degree('out')
            deg_cent = self.centrality('degree')
            in_deg_cent = self.centrality('in_degree')
            # bet_cent = self.betweenness_centrality()
            eig_cent = self.centrality('eigenvector')
            cls_cent = self.centrality('closeness')
            
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
