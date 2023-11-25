import time
import networkx as nx
import numpy as np
import json
from collections import Counter
import shutil
from pathlib import Path

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.utils.types import (
    NodeList,
    EdgeList,
    EdgeAttribute,
    NodeAttribute,
    EdgeAttributeKind,
    NodeAttributeKind,
    ComponentKind,
    ReachableKind,
    CentralityKind,
    DegreeKind,
    ShortestPathKind,
    CommunityKind,
)

from .components import get_components
from .degree import compute_degree
from .centrality import compute_centrality
from .edge_attributes import get_edge_attribute
from .node_attributes import get_node_attribute
from .reachable import get_reachable
from .shortest_path import get_shortest_path
from .communities import get_communities

class CryptoChatterGraph:
    id: str
    source: int|None=None
    G: nx.Graph
    nodes: NodeList
    edges: EdgeList
    cache_dir: Path

    def __init__(
        self,
        _id: str,
        G: nx.Graph,
        nodes: NodeList,
        edges: EdgeList,
        cache_dir: Path,
        source: int|None=None,
    ) -> None:
        self.id = _id
        self.G = G
        self.nodes = nodes
        self.edges = edges
        self.cache_dir = cache_dir / self.id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.source = None

    def clear_cache(self) -> None:
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def degree(
        self,
        kind: DegreeKind,
    ) -> np.ndarray:
        save_file = self.cache_dir / f"stats/degree/{kind}.npy"
        save_file.parent.mkdir(exist_ok=True, parents=True)
        if not save_file.is_file():
            degree = compute_degree(
                G=self.G,
                nodes=self.nodes,
                # graph_config=self.graph_config,
                kind=kind,
            )
            np.save(open(save_file, "wb"), degree)
        else:
            degree = np.load(open(save_file, "rb"))
        return degree

    def centrality(self, kind: CentralityKind) -> np.ndarray:
        save_file = self.cache_dir / f"stats/centrality/{kind}.npy"
        save_file.parent.mkdir(exist_ok=True, parents=True)
        try:
            with open(save_file, "rb") as f:
                centrality = np.load(f)
        except (FileNotFoundError):
            centrality = compute_centrality(G=self.G, nodes=self.nodes, kind=kind)
            with open(save_file, "wb") as f:
                np.save(f, centrality)
        return centrality

    def reachable(
        self,
        node: int,
        kind: ReachableKind,
    ) -> dict[int,NodeList]:
        reachable_nodes_file = self.cache_dir / f"reachable/{kind}/{node}.json"
        reachable_nodes_file.parent.mkdir(parents=True, exist_ok=True)
        node = int(node)

        try:
            with open(reachable_nodes_file, "r") as f:
                reachable_nodes = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            reachable_nodes = get_reachable(
                G=self.G,
                node=node,
                kind=kind,
            )
            with open(reachable_nodes_file, "w") as f:
                json.dump(reachable_nodes, f)
        return reachable_nodes

    def components(
        self,
        kind: ComponentKind,
    ) -> list[NodeList]:
        components_file = self.cache_dir / f"components/{kind}.json"
        components_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(components_file, "r") as f:
                components = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(components_file, "w") as f:
                components = get_components(
                    G=self.G,
                    component_kind=kind,
                )
                json.dump(components, f)
        return components


    def communities(
        self, 
        kind: CommunityKind,
        random_seed: int,
    ) -> list[NodeList]:
        save_file = self.cache_dir / f"communities/{kind}_{random_seed}.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_file, "r") as f:
                communities = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(save_file, "w") as f:
                communities = get_communities(
                    G=self.G,
                    community_kind=kind,
                    random_seed=random_seed,
                )
                json.dump(communities, f)
        return communities

    def diameter(self) -> int:
        save_file = self.cache_dir / "stats/diameter.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_file, "r") as f:
                diameter = json.load(f)['diameter']
        except (FileNotFoundError, json.JSONDecodeError):
            with open(save_file, "w") as f:
                diameter = nx.diameter(self.G)
                json.dump({'diameter': diameter}, f)
        return diameter

    def get_keywords(
        self,
        data: CryptoChatterData,
    ) -> dict[str, float]:
        save_file = self.cache_dir / f"stats/keywords/{data.tfidf_config}.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_file, "r") as f:
                keywords_with_score = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(save_file, "w") as f:
                keywords_with_score = data.get_tfidf(
                    texts=data.get("clean_text", self.nodes)
                )
                json.dump(keywords_with_score, f)
        return keywords_with_score

    def count_hashtags(
        self,
        data: CryptoChatterData,
        top_n: int = 100,
    ) -> dict[str, int]:
        save_file = self.cache_dir / f"stats/hashtags.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_file, "r") as f:
                hashtag_count = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(save_file, "w") as f:
                hashtag_count = dict(
                    Counter(
                        [
                            tag
                            for hashtags in data.get("hashtags", self.nodes)
                            for tag in hashtags
                        ]
                    ).most_common()[:top_n]
                )
                json.dump(hashtag_count, f)
        return hashtag_count

    def get_node_attribute(
        self,
        data: CryptoChatterData,
        kind: NodeAttributeKind,
    ) -> NodeAttribute:
        node_attr_file = self.cache_dir / f"node_attributes/{kind}.json"
        node_attr_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(node_attr_file, "r") as f:
                node_attr = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(node_attr_file, "w") as f:
                node_attr = get_node_attribute(nodes=self.nodes, data=data, kind=kind)
                json.dump(node_attr, f)
        return node_attr

    def get_edge_attribute(
        self,
        data: CryptoChatterData,
        kind: EdgeAttributeKind,
    ) -> EdgeAttribute:
        edge_attr_file = self.cache_dir / f"edge_attributes/{kind}.json"
        edge_attr_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(edge_attr_file, "r") as f:
                edge_attr = json.load(f)
                edge_attr = dict(
                    zip(
                        [tuple(e) for e in edge_attr["keys"]],
                        edge_attr["values"],
                    )
                )
        except (FileNotFoundError, json.JSONDecodeError):
            with open(edge_attr_file, "w") as f:
                edge_attr = get_edge_attribute(edges=self.edges, data=data, kind=kind)
                json.dump({
                    "keys": list(edge_attr.keys()),
                    "values": list(edge_attr.values())
                }, f)
        return edge_attr

    def export_gephi(
        self,
        data: CryptoChatterData,
        node_attributes: list[NodeAttributeKind] = [],
        edge_attributes: list[EdgeAttributeKind] = [],
    ) -> None:
        start = time.time()
        for attr in node_attributes:
            nx.set_node_attributes(
                G=self.G,
                values=self.get_node_attribute(data=data, kind=attr),
                name=attr,
            )
        for attr in edge_attributes:
            nx.set_edge_attributes(
                G=self.G,
                values=self.get_edge_attribute(data=data, kind=attr),
                name=attr,
            )
        print(f"exported to gephi graph in {time.time()-start:.2f} seconds")
        nx.write_gexf(self.G, self.cache_dir / "graph.gexf")
