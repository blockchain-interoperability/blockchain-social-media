import time
import json
from rich.progress import Progress

from crypto_chatter.utils.types import (
    ComponentKind,
    SubGraphKind,
    ReachableKind,
    CentralityKind,
)

from .reachable import get_reachable
from .graph import CryptoChatterGraph
from .sub_graph import CryptoChatterSubGraph
from .components import get_components


class CryptoChatterSubGraphBuilder:
    graph: CryptoChatterGraph
    progress: Progress|None
    use_progress: bool = False

    def __init__(
        self,
        graph: CryptoChatterGraph,
        progress: Progress|None = None,
    ):
        self.graph = graph
        self.progress = progress
        self.use_progress = progress is not None
    
    def get_subgraphs(
        self,
        subgraph_kind: SubGraphKind,
        top_n: int = 10,
        **kwargs
    ) -> list[CryptoChatterSubGraph]:
        if subgraph_kind == "centrality":
            centrality_kind = kwargs.get("centrality")
            reachable_kind = kwargs.get("reachable")
            return self.get_subgraphs_centrality(
                centrality_kind=centrality_kind,
                reachable_kind=reachable_kind,
                top_n=top_n,
            )
        elif subgraph_kind == "component":
            component_kind = kwargs.get("component")
            return self.get_subgraphs_components(
                top_n=top_n,
                component_kind=component_kind,
            )
        else:
            raise NotImplementedError(f"{subgraph_kind} subgraph kind is not implemented!")

    def get_subgraphs_components(
        self,
        component_kind: ComponentKind,
        top_n: int,
    ) -> list[CryptoChatterSubGraph]:
        start = time.time()
        components_file = self.graph.graph_config.graph_dir / f'components/{component_kind}.json'
        components_file.parent.mkdir(parents=True, exist_ok=True)
        if not components_file.is_file():
            components = get_components(
                self.graph.G,
                component_kind=component_kind,
            )
            json.dump(components, open(components_file, "w"))
        else:
            components = json.load(open(components_file))

        if self.use_progress:
            progress_task = self.progress.add_task(
                description=f"loading subgraphs by wcc..",
                total=top_n,
            )

        subgraphs = []
        for i,component in enumerate(components[:top_n]):
            subgraph_id = f"components/{component_kind}/{i}"
            subgraphs += [
                CryptoChatterSubGraph(
                    subgraph_id,
                    parent=self.graph,
                    nodes=list(component),
                )
            ]
            if self.use_progress:
                self.progress.advance(progress_task)

        if self.use_progress:
            self.progress.remove_task(progress_task)

        print(f"loaded component [{component_kind}] subgraphs in {time.time() - start:.2f} seconds")
        return subgraphs

    def get_subgraphs_centrality(
        self,
        centrality_kind: CentralityKind,
        reachable_kind: ReachableKind,
        top_n: int,
    ) -> list[CryptoChatterSubGraph]:
        start = time.time()
        if self.use_progress:
            subgraph_task = self.progress.add_task(
                description=f"loading subgraphs by {centrality_kind}-{reachable_kind}..",
                total=top_n,
            )

        subgraphs = []
        centrality_idx = self.graph.centrality(centrality_kind).argsort()[::-1]
        for i, idx in enumerate(centrality_idx[:top_n]):
            node = self.graph.nodes[idx]
            reachable_nodes_file = self.graph.graph_config.graph_dir / f'reachable/{reachable_kind}/{node}.json'
            reachable_nodes_file.parent.mkdir(parents=True, exist_ok=True)

            if not reachable_nodes_file.is_file():
                reachable_nodes = self.graph.reachable(
                    node = node,
                    kind = reachable_kind,
                )
                json.dump(reachable_nodes, open(reachable_nodes_file, "w"))
            else:
                reachable_nodes = json.load(open(reachable_nodes_file))

            subgraph_id = f"centrality/{centrality_kind}-{reachable_kind}/{i}-{int(node)}"
            subgraphs += [
                CryptoChatterSubGraph(
                    subgraph_id,
                    parent=self.graph,
                    nodes=reachable_nodes,
                )
            ]

            if self.use_progress:
                self.progress.advance(subgraph_task)

        if self.use_progress:
            self.progress.remove_task(subgraph_task)

        print(f"loaded {centrality_kind}-{reachable_kind} subgraphs in {time.time() - start:.2f} seconds")
        return subgraphs
