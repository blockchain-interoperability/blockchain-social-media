import networkx as nx
from networkx.readwrite.gml import re
from rich.progress import Progress

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.config import CryptoChatterGraphConfig
from crypto_chatter.utils.types import (
    ComponentKind,
    CommunityKind,
    SubGraphKind,
    ReachableKind,
    CentralityKind,
)

from .graph import CryptoChatterGraph
from .build_graph import build_graph

class CryptoChatterGraphBuilder:
    data: CryptoChatterData
    graph_config: CryptoChatterGraphConfig
    progress: Progress | None
    use_progress: bool = False

    def __init__(
        self,
        data: CryptoChatterData,
        graph_config: CryptoChatterGraphConfig,
        progress: Progress | None = None,
    ):
        self.data = data
        self.progress = progress
        self.use_progress = progress is not None
        self.graph_config = graph_config

    def get_graph(
        self,
    ):
        nodes, edges = build_graph(
            data=self.data,
            graph_config=self.graph_config,
        )

        return CryptoChatterGraph(
            _id="main",
            G=nx.DiGraph(edges),
            nodes=nodes,
            edges=edges,
            cache_dir=self.graph_config.graph_dir,
        )

    def get_subgraphs(
        self,
        graph: CryptoChatterGraph,
        kind: SubGraphKind,
        top_n: int = 10,
        **kwargs,
    ) -> list[CryptoChatterGraph]:
        if kind == "centrality":
            centrality_kind = kwargs.get("centrality")
            reachable_kind = kwargs.get("reachable")
            return self.get_subgraphs_centrality(
                graph=graph,
                centrality_kind=centrality_kind,
                reachable_kind=reachable_kind,
                top_n=top_n,
            )
        elif kind == "component":
            component_kind = kwargs.get("component")
            return self.get_subgraphs_components(
                graph=graph,
                top_n=top_n,
                component_kind=component_kind,
            )
        elif kind == "community":
            community_kind = kwargs.get("community")
            random_seed = kwargs.get("random_seed", 0)
            return self.get_subgraphs_communities(
                graph=graph,
                top_n=top_n,
                community_kind=community_kind,
                random_seed=random_seed,
            )
        else:
            raise NotImplementedError(
                f"{subgraph_kind} subgraph kind is not implemented!"
            )

    def get_subgraphs_centrality(
        self,
        graph: CryptoChatterGraph,
        centrality_kind: CentralityKind,
        reachable_kind: ReachableKind,
        top_n: int,
    ) -> list[CryptoChatterGraph]:
        if self.use_progress:
            subgraph_task = self.progress.add_task(
                description=f"loading subgraphs by {centrality_kind}-{reachable_kind}..",
                total=top_n,
            )

        subgraphs = []
        centrality_idx = graph.centrality(centrality_kind).argsort()[::-1]
        for i, idx in enumerate(centrality_idx[:top_n]):
            node = graph.nodes[idx]
            reachable_nodes = list(graph.reachable(
                node=node,
                kind=reachable_kind,
            ).keys())

            subgraph_id = f"subgraph/centrality/{centrality_kind}-{reachable_kind}/{i}-{int(node)}"
            G = graph.G.subgraph(reachable_nodes)
            edges = list(graph.G.edges(reachable_nodes))
            subgraphs += [
                CryptoChatterGraph(
                    _id=subgraph_id,
                    G=G,
                    nodes=reachable_nodes,
                    edges=edges,
                    cache_dir=self.graph_config.graph_dir,
                    # source=node,
                )
            ]

            if self.use_progress:
                self.progress.advance(subgraph_task)

        if self.use_progress:
            self.progress.remove_task(subgraph_task)

        return subgraphs

    def get_subgraphs_components(
        self,
        graph: CryptoChatterGraph,
        component_kind: ComponentKind,
        top_n: int,
    ) -> list[CryptoChatterGraph]:
        components = graph.components(component_kind)
        if self.use_progress:
            progress_task = self.progress.add_task(
                description=f"loading subgraphs by {component_kind} component..",
                total=top_n,
            )

        subgraphs = []
        for i, component in enumerate(components[:top_n]):
            subgraph_id = f"subgraph/component/{component_kind}/{i}"
            G = graph.G.subgraph(component)
            edges = list(graph.G.edges(component))
            subgraphs += [
                CryptoChatterGraph(
                    _id=subgraph_id,
                    G=G,
                    nodes=component,
                    edges=edges,
                    cache_dir=self.graph_config.graph_dir,
                )
            ]
            if self.use_progress:
                self.progress.advance(progress_task)

        if self.use_progress:
            self.progress.remove_task(progress_task)

        return subgraphs

    def get_subgraphs_communities(
        self,
        graph: CryptoChatterGraph,
        top_n: int,
        community_kind: CommunityKind,
        random_seed: int=0,
    ):
        communities = graph.communities(
            kind=community_kind,
            random_seed=random_seed,
        )

        if self.use_progress:
            progress_task = self.progress.add_task(
                description=f"loading subgraphs by {community_kind} community..",
                total=top_n,
            )

        subgraphs = []
        for i,community in enumerate(communities[:top_n]):
            subgraph_id = f"subgraph/community/{community_kind}_{random_seed}/{i}"
            G = graph.G.subgraph(community)
            edges = list(graph.G.edges(community))
            subgraphs += [
                CryptoChatterGraph(
                    _id=subgraph_id,
                    G=G,
                    nodes=community,
                    edges=edges,
                    cache_dir=self.graph_config.graph_dir,
                )
            ]

            if self.use_progress:
                self.progress.advance(progress_task)

        if self.use_progress:
            self.progress.remove_task(progress_task)

        return subgraphs
