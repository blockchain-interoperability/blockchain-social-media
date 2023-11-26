import time
import pandas as pd
import networkx as nx
import numpy as np
import json
from collections import Counter
import shutil
from pathlib import Path
from rich.progress import Progress

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
    CommunityKind,
)

from .components import get_components
from .degree import compute_degree
from .centrality import compute_centrality
from .edge_attributes import get_edge_attribute
from .node_attributes import get_node_attribute
from .reachable import get_reachable
from .communities import get_communities


class CryptoChatterGraph:
    id: str
    source: int | None = None
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

    # def diameter(self) -> int:
    #     save_file = self.cache_dir / "stats/diameter.json"
    #     save_file.parent.mkdir(parents=True, exist_ok=True)
    #     try:
    #         with open(save_file, "r") as f:
    #             diameter = json.load(f)['diameter']
    #     except (FileNotFoundError, json.JSONDecodeError):
    #         with open(save_file, "w") as f:
    #             diameter = nx.diameter(self.G)
    #             json.dump({'diameter': diameter}, f)
    #     return diameter

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
                edge_attr = get_edge_attribute(
                    edges=self.edges,
                    data=data,
                    kind=kind,
                )
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

    def stats(
        self,
        data: CryptoChatterData,
        include_edge: bool = False,
        include_keywords: bool = False,
        recompute: bool = False,
        progress: Progress | None = None,
    ) -> str:
        stats_file = self.cache_dir / "stats/stats.json"
        edge_stats_file = self.cache_dir / "stats/edge_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        output = ""
        if progress is not None:
            num_task = 1 + int(include_edge) + int(include_keywords)
            task = progress.add_task("Calculating stats", total=num_task)

        if not stats_file.is_file() or recompute:
            stats = [
                {"Type": "Node Count", "Val": len(self.nodes)},
                {"Type": "Edge Count", "Val": len(self.edges)},
            ]

            undir = self.G.to_undirected()
            if nx.is_connected(undir):
                print("getting undir diam")
                stats += [
                    {"Type": "Undirected Diamater", "Val": nx.diameter(undir)},
                ]
            else:
                stats += [
                    {"Type": "Undirected Diamater", "Val": -1},
                ]

            if self.G.is_directed() and nx.is_strongly_connected(self.G):
                print("getting dir diam")
                stats += [
                    {"Type": "Directed Diamater", "Val": nx.diameter(self.G)},
                ]
            else:
                stats += [
                    {"Type": "Directed Diamater", "Val": -1},
                ]

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

            json.dump(stats, open(stats_file, "w"))

        else:
            stats = json.load(open(stats_file, "r"))
        stats = pd.DataFrame(stats)
        output += "**GRAPH STATS**\n"
        output += stats.to_markdown(index=False, floatfmt=".0f")
        output += "\n\n"

        if progress is not None:
            progress.advance(task)

        if include_edge:
            if not edge_stats_file.is_file() or recompute:
                edge_sims = pd.DataFrame(
                    [
                        dict(
                            node_from=n1,
                            node_to=n2,
                            sim=sim,
                        )
                        for (n1, n2), sim in self.get_edge_attribute(
                            data=data, kind="emb_cosine_sim"
                        ).items()
                    ]
                )

                edge_stats = [
                    {"Type": "Edge Sim Average", "Val": edge_sims["sim"].mean()},
                    {"Type": "Edge Sim STD", "Val": edge_sims["sim"].std()},
                    {"Type": "Edge Sim Max", "Val": int(edge_sims["sim"].max())},
                    {"Type": "Edge Sim Min", "Val": int(edge_sims["sim"].min())},
                ]
                json.dump(edge_stats, open(edge_stats_file, "w"))
            else:
                edge_stats = json.load(open(edge_stats_file, "r"))
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
                )
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
