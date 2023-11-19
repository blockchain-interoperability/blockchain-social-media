import networkx as nx

from crypto_chatter.utils.types import (
    NodeList,
    ComponentKind,
)

def get_weakly_connected_components(
    G: nx.Graph
) -> list[NodeList]:
    return sorted(
        [list(cc) for cc in nx.weakly_connected_components(G)],
        key=len,
        reverse=True
    )

def get_strongly_connected_components(
    G: nx.Graph
) -> list[NodeList]:
    if not isinstance(G, nx.DiGraph):
        raise ValueError(f"graph must be undirected to compute strongly connected components")
    return sorted(
        [list(cc) for cc in nx.strongly_connected_components(G)],
        key=len,
        reverse=True
    )

component_functions = {
    'weak': get_weakly_connected_components,
    'strong': get_strongly_connected_components,
}

def get_components(
    G: nx.Graph,
    component_kind: ComponentKind,
) -> list[NodeList]:
    if component_kind not in component_functions:
        raise ValueError(f"invalid components kind: [{component_kind}]")
    return component_functions[component_kind](G)
