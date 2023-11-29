import time
import pandas as pd
import networkx as nx
import numpy as np
import json
from collections import Counter
import shutil
from pathlib import Path
from itertools import chain
from rich.progress import Progress

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.utils import unslug
from crypto_chatter.utils.types import (
    GraphKind,
    NodeList,
    NodeToIdMap,
    EdgeList,
    EdgeAttribute,
    NodeAttribute,
    TweetGraphEdgeAttributeKind,
    TweetGraphNodeAttributeKind,
    UserGraphEdgeAttributeKind,
    UserGraphNodeAttributeKind,
    ComponentKind,
    ReachableKind,
    CentralityKind,
    DegreeKind,
    DiameterKind,
    CommunityKind,
)

from .components import get_components
from .degree import compute_degree
from .diameter import get_diameter
from .centrality import compute_centrality
from .reachable import get_reachable
from .communities import get_communities
from . import node_attribute
from . import edge_attribute


class CryptoChatterGraph:
    id: str
    source: int | None = None
    G: nx.Graph
    nodes: NodeList
    node_to_ids: NodeToIdMap
    edges: EdgeList
    cache_dir: Path
    kind: GraphKind

    def __init__(
        self,
        _id: str,
        G: nx.Graph,
        nodes: NodeList,
        node_to_ids: NodeToIdMap,
        edges: EdgeList,
        kind: GraphKind,
        cache_dir: Path,
    ) -> None:
        self.id = _id
        self.G = G
        self.nodes = nodes
        self.node_to_ids = node_to_ids
        self.edges = edges
        self.kind = kind
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

    def diamater(
        self,
        kind: DiameterKind,
    ):
        save_file = self.cache_dir / f"stats/diameter/{kind}.json"
        save_file.parent.mkdir(exist_ok=True, parents=True)
        if not save_file.is_file():
            diameter = get_diameter(G=self.G, kind=kind)
            json.dump(diameter, open(save_file, "w"))
        else:
            diameter = json.load(open(save_file, "r"))
        return diameter

    def centrality(self, kind: CentralityKind) -> np.ndarray:
        save_file = self.cache_dir / f"stats/centrality/{kind}.npy"
        save_file.parent.mkdir(exist_ok=True, parents=True)
        try:
            with open(save_file, "rb") as f:
                centrality = np.load(f)
        except FileNotFoundError:
            centrality = compute_centrality(G=self.G, nodes=self.nodes, kind=kind)
            with open(save_file, "wb") as f:
                np.save(f, centrality)
        return centrality

    def reachable(
        self,
        node: int,
        kind: ReachableKind,
    ) -> dict[int, NodeList]:
        reachable_nodes_file = self.cache_dir / f"reachable/{kind}/{node}.json"
        reachable_nodes_file.parent.mkdir(parents=True, exist_ok=True)
        node = int(node)

        try:
            with open(reachable_nodes_file, "r") as f:
                reachable_nodes = {int(k): v for k, v in json.load(f).items()}

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
            node_ids = list(chain.from_iterable(self.node_to_ids.values()))
            keywords_with_score = data.get_tfidf(texts=data.get("clean_text", node_ids))
            with open(save_file, "w") as f:
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
            node_ids = list(chain.from_iterable(self.node_to_ids.values()))
            hashtag_count = dict(
                Counter(
                    [
                        tag
                        for hashtags in data.get("hashtags", node_ids)
                        for tag in hashtags
                    ]
                ).most_common()[:top_n]
            )
            with open(save_file, "w") as f:
                json.dump(hashtag_count, f)
        return hashtag_count

    def get_node_attribute(
        self,
        data: CryptoChatterData,
        kind: TweetGraphNodeAttributeKind | UserGraphNodeAttributeKind,
        progress: Progress | None = None,
    ) -> NodeAttribute:
        node_attr_file = self.cache_dir / f"node_attributes/{kind}.json"
        node_attr_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(node_attr_file, "r") as f:
                node_attr = json.load(f)
                node_attr = {int(k): v for k, v in node_attr.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            if "tweet" in self.kind:
                node_attr = node_attribute.twitter.get_tweet_node_attribute(
                    nodes=self.nodes,
                    data=data,
                    kind=kind,
                )
            elif "user" in self.kind:
                node_attr = node_attribute.twitter.get_user_node_attribute(
                    nodes=self.nodes,
                    node_to_ids=self.node_to_ids,
                    data=data,
                    kind=kind,
                    progress=progress,
                )
            else:
                raise ValueError(f"Node attributes not implemented for: {self.kind}")
            with open(node_attr_file, "w") as f:
                json.dump(node_attr, f)
        return node_attr

    def get_edge_attribute(
        self,
        data: CryptoChatterData,
        kind: TweetGraphEdgeAttributeKind | UserGraphEdgeAttributeKind,
        progress: Progress | None = None,
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
            if "tweet" in self.kind:
                edge_attr = edge_attribute.twitter.get_tweet_edge_attribute(
                    edges=self.edges,
                    data=data,
                    kind=kind,
                )
            elif "user" in self.kind:
                edge_attr = edge_attribute.twitter.get_user_edge_attribute(
                    edges=self.edges,
                    node_to_ids=self.node_to_ids,
                    data=data,
                    kind=kind,
                    progress=progress,
                )
            else:
                raise ValueError(f"Edge attributes not implemented for: {self.kind}")
            with open(edge_attr_file, "w") as f:
                json.dump(
                    {
                        "keys": list(edge_attr.keys()),
                        "values": list(edge_attr.values()),
                    },
                    f,
                )
        return edge_attr

    def export_gephi(
        self,
        data: CryptoChatterData,
        node_attributes: list[
            TweetGraphNodeAttributeKind | UserGraphNodeAttributeKind
        ] = [],
        edge_attributes: list[
            TweetGraphEdgeAttributeKind | UserGraphEdgeAttributeKind
        ] = [],
        progress: Progress | None = None,
    ) -> None:
        start = time.time()
        if progress is not None:
            task = progress.add_task(
                "Exporting to gephi graph",
                total=1 + len(edge_attributes) + len(node_attributes),
            )
        for attr in node_attributes:
            nx.set_node_attributes(
                G=self.G,
                values=self.get_node_attribute(
                    data=data,
                    kind=attr,
                    progress=progress,
                ),
                name=unslug(attr),
            )
            if progress is not None:
                progress.advance(task)
        for attr in edge_attributes:
            nx.set_edge_attributes(
                G=self.G,
                values=self.get_edge_attribute(
                    data=data,
                    kind=attr,
                    progress=progress,
                ),
                name=unslug(attr),
            )
            if progress is not None:
                progress.advance(task)

        print(f"exported to gephi graph in {time.time()-start:.2f} seconds")
        nx.write_gexf(self.G, self.cache_dir / f"graph.gexf")
        if progress is not None:
            progress.advance(task)
            progress.remove_task(task)

    def stats(
        self,
        data: CryptoChatterData,
        node_attributes: list[
            TweetGraphNodeAttributeKind | UserGraphNodeAttributeKind
        ] = [],
        edge_attributes: list[
            TweetGraphEdgeAttributeKind | UserGraphEdgeAttributeKind
        ] = [],
        include_keywords: bool = False,
        top_n_keywords: int = 10,
        progress: Progress | None = None,
    ) -> str:
        output = ""
        if progress is not None:
            num_task = (
                1 + len(edge_attributes) + len(node_attributes) + int(include_keywords)
            )
            task = progress.add_task("Calculating stats", total=num_task)

        stats = [
            {"Type": "Node Count", "Val": len(self.G.nodes())},
            {"Type": "Edge Count", "Val": len(self.G.edges())},
        ]

        # stats += [ {"Type": "Diameter", "Val": self.diamater('undirected')} ]
        # if self.G.is_directed():
        #     stats += [ {"Type": "Directed Diameter", "Val": self.diamater('directed')} ]

        deg = self.degree(kind="all")
        stats += [
            {"Type": "Degree Average", "Val": np.mean(deg)},
            {"Type": "Degree Median", "Val": int(np.median(deg))},
            {"Type": "Degree STD", "Val": np.std(deg)},
            {"Type": "Degree Max", "Val": int(np.max(deg))},
            {"Type": "Degree Min", "Val": int(np.min(deg))},
        ]

        if self.G.is_directed():
            in_deg = self.degree(kind="in")
            stats += [
                {"Type": "In Degree Average", "Val": np.mean(in_deg)},
                {"Type": "In Degree Median", "Val": int(np.median(in_deg))},
                {"Type": "In Degree STD", "Val": np.std(in_deg)},
                {"Type": "In Degree Max", "Val": int(np.max(in_deg))},
                {"Type": "In Degree Min", "Val": int(np.min(in_deg))},
            ]

            out_deg = self.degree(kind="out")
            stats += [
                {"Type": "Out Degree Average", "Val": np.mean(out_deg)},
                {"Type": "Out Degree Median", "Val": int(np.median(out_deg))},
                {"Type": "Out Degree STD", "Val": np.std(out_deg)},
                {"Type": "Out Degree Max", "Val": int(np.max(out_deg))},
                {"Type": "Out Degree Min", "Val": int(np.min(out_deg))},
            ]

        stats = pd.DataFrame(stats)
        output += "**GRAPH STATS**\n"
        output += stats.to_markdown(index=False, floatfmt=".0f")
        output += "\n\n"

        if progress is not None:
            progress.advance(task)

        node_stats = []
        for node_attr in node_attributes:
            # print(node_attr)
            if node_attr == "text":
                pass
            node_val = pd.DataFrame(
                [
                    dict(
                        node=n,
                        value=sim,
                    )
                    for n, sim in self.get_node_attribute(
                        data=data,
                        kind=node_attr,
                        progress=progress,
                    ).items()
                ]
            )

            node_stats += [
                {
                    "Type": f"Node {unslug(node_attr)} Average",
                    "Val": node_val["value"].mean(),
                },
                {
                    "Type": f"Node {unslug(node_attr)} STD",
                    "Val": node_val["value"].std(),
                },
                {
                    "Type": f"Node {unslug(node_attr)} Max",
                    "Val": int(node_val["value"].max()),
                },
                {
                    "Type": f"Node {unslug(node_attr)} Min",
                    "Val": int(node_val["value"].min()),
                },
            ]

        if len(node_attributes) > 0:
            node_stats = pd.DataFrame(node_stats)
            output += "**NODE STATS**\n"
            output += node_stats.to_markdown(index=False, floatfmt=".4f")
            output += "\n\n"
            if progress is not None:
                progress.advance(task)

        # print("converted node stats to markdown.. now edges")
        edge_stats = []
        for edge_attr in edge_attributes:
            # print(edge_attr)
            edge_val = pd.DataFrame(
                [
                    dict(
                        node_from=n1,
                        node_to=n2,
                        value=sim,
                    )
                    for (n1, n2), sim in self.get_edge_attribute(
                        data=data,
                        kind=edge_attr,
                        progress=progress,
                    ).items()
                ]
            )
            # print("edge_val", edge_val.shape, edge_val.columns)

            edge_stats += [
                {
                    "Type": f"Edge {unslug(edge_attr)} Average",
                    "Val": edge_val["value"].mean(),
                },
                {
                    "Type": f"Edge {unslug(edge_attr)} STD",
                    "Val": edge_val["value"].std(),
                },
                {
                    "Type": f"Edge {unslug(edge_attr)} Max",
                    "Val": int(edge_val["value"].max()),
                },
                {
                    "Type": f"Edge {unslug(edge_attr)} Min",
                    "Val": int(edge_val["value"].min()),
                },
            ]

        if len(edge_attributes) > 0:
            edge_stats = pd.DataFrame(edge_stats)
            output += "**EDGE STATS**\n"
            output += edge_stats.to_markdown(index=False, floatfmt=".4f")
            output += "\n\n"
            if progress is not None:
                progress.advance(task)

        if include_keywords:
            keywords = [
                {"Keyword": kw, "Score": sc}
                for kw, sc in self.get_keywords(
                    data=data,
                )[:top_n_keywords]
            ]
            keywords = pd.DataFrame(keywords)
            output += "**KEYWORDS**\n"
            output += keywords.to_markdown(index=False, floatfmt=".2f")
            output += "\n\n"

            if progress is not None:
                progress.advance(task)

        if progress is not None:
            progress.remove_task(task)

        return output
