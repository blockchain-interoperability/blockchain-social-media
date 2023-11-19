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
from .graph import CryptoChatterGraph

class CryptoChatterSubGraph:
    id: str
    parent: CryptoChatterGraph
    nodes: NodeList
    edges: EdgeList
    graph: nx.Graph
    cache_dir: Path

    def __init__(
        self, 
        _id: str,
        parent: "CryptoChatterGraph", 
        nodes: NodeList,
    ):
        self.id = _id
        self.parent = parent
        self.cache_dir = self.parent.graph_config.graph_dir / f"subgraph/{_id}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.nodes = nodes
        self.G = self.parent.G.subgraph(self.nodes)
        self.edges = self.G.edges(self.nodes)

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
            vecs = self.parent.data.tfidf.transform(self.parent.data.get('text', self.nodes_in_data))
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
                    for hashtags in self.parent.data.get("hashtags", self.nodes_in_data)
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
        start=time.time()
        for attr in node_attributes:
            nx.set_node_attributes(
                G=self.G,
                values=get_node_attribute(
                    nodes=self.nodes_in_data,
                    data=self.parent.data,
                    kind=attr
                ),
                name=attr
            )
        for attr in edge_attributes:
            nx.set_edge_attributes(
                G=self.G,
                values=get_edge_attribute(
                    edges=self.edges_in_data,
                    data=self.parent.data,
                    kind=attr
                ),
                name=attr
            )
        print(f"exported to gephi graph in {time.time()-start:.2f} seconds")
        nx.write_gexf(self.G, self.cache_dir / "graph.gexf")

