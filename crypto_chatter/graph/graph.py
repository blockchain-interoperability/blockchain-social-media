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
)

from .components import get_components
from .degree import compute_degree
from .centrality import compute_centrality
from .edge_attributes import get_edge_attribute
from .node_attributes import get_node_attribute
from .reachable import get_reachable
from .shortest_path import get_shortest_path

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
        if not save_file.is_file():
            centrality = compute_centrality(G=self.G, nodes=self.nodes, kind=kind)
            np.save(open(save_file, "wb"), centrality)
        else:
            centrality = np.load(open(save_file, "rb"))
        return centrality

    def reachable(
        self,
        node: int,
        kind: ReachableKind,
    ) -> NodeList:
        reachable_nodes_file = self.cache_dir / f"reachable/{kind}/{node}.json"
        reachable_nodes_file.parent.mkdir(parents=True, exist_ok=True)
        if not reachable_nodes_file.is_file():
            reachable_nodes = get_reachable(
                G=self.G,
                node=node,
                kind=kind,
            )
            json.dump(reachable_nodes, open(reachable_nodes_file, "w"))
        else:
            reachable_nodes = json.load(open(reachable_nodes_file))
        return reachable_nodes

    def shortest_path(
        self,
        source: int, 
        kind: ShortestPathKind,
    ):
        shortest_path_file = self.cache_dir / f"shortest_path/{kind}.json"
        shortest_path_file.parent.mkdir(parents=True, exist_ok=True)
        if not shortest_path_file.is_file():
            shortest_path = get_shortest_path(
                G=self.G,
                source=source,
                nodes=self.nodes,
                kind=kind,
            )
            json.dump(shortest_path, open(shortest_path_file, "w"))
        else:
            shortest_path = json.load(open(shortest_path_file))
        return shortest_path

    def components(
        self,
        component_kind: ComponentKind,
    ) -> list[NodeList]:
        components_file = self.cache_dir / f"components/{component_kind}.json"
        components_file.parent.mkdir(parents=True, exist_ok=True)
        if not components_file.is_file():
            components = get_components(
                self.G,
                component_kind=component_kind,
            )
            json.dump(components, open(components_file, "w"))
        else:
            components = json.load(open(components_file))
        return components

    def diameter(self) -> int:
        save_file = self.cache_dir / "stats/diameter.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        if not save_file.is_file():
            diameter = nx.diameter(self.G)
            json.dump({'diameter': diameter}, open(save_file, "w"))
        else:
            diameter = json.load(open(save_file))['diameter']
        return diameter

    def get_keywords(
        self,
        data: CryptoChatterData,
    ) -> dict[str, float]:
        save_file = self.cache_dir / f"stats/keywords/{data.tfidf_config}.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        if not save_file.is_file():
            keywords_with_score = data.get_tfidf(
                    texts=data.get("text", self.nodes)
                )

            json.dump(keywords_with_score, open(save_file, "w"))
        else:
            keywords_with_score = json.load(open(save_file))

        return keywords_with_score

    def count_hashtags(
        self,
        data: CryptoChatterData,
        top_n: int = 100,
    ) -> dict[str, int]:
        save_file = self.cache_dir / f"stats/hashtags.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        if not save_file.is_file():
            hashtag_count = dict(
                Counter(
                    [
                        tag
                        for hashtags in data.get("hashtags", self.nodes)
                        for tag in hashtags
                    ]
                ).most_common()[:top_n]
            )
            json.dump(hashtag_count, open(save_file, "w"))
        else:
            hashtag_count = json.load(open(save_file))
        return hashtag_count

    def get_node_attribute(
        self,
        data: CryptoChatterData,
        kind: NodeAttributeKind,
    ) -> NodeAttribute:
        node_attr_file = self.cache_dir / f"node_attributes/{kind}.json"
        node_attr_file.parent.mkdir(parents=True, exist_ok=True)
        if not node_attr_file.is_file():
            node_attr = get_node_attribute(nodes=self.nodes, data=data, kind=kind)
            json.dump(node_attr, open(node_attr_file, "w"))
        else:
            node_attr = json.load(open(node_attr_file))
        return node_attr

    def get_edge_attribute(
        self,
        data: CryptoChatterData,
        kind: EdgeAttributeKind,
    ) -> EdgeAttribute:
        edge_attr_file = self.cache_dir / f"edge_attributes/{kind}.json"
        edge_attr_file.parent.mkdir(parents=True, exist_ok=True)
        if not edge_attr_file.is_file():
            edge_attr = get_edge_attribute(edges=self.edges, data=data, kind=kind)
            json.dump(
                {
                    "keys": list(edge_attr.keys()),
                    "values": list(edge_attr.values())
                }, 
                open(edge_attr_file, "w")
            )
        else:
            edge_attr = json.load(open(edge_attr_file))
            edge_attr = dict(
                zip(
                    [tuple(e) for e in edge_attr["keys"]],
                    edge_attr["values"],
                )
            )
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
