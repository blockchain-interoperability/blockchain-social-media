import networkx as nx

from crypto_chatter.utils.types import (
    NodeList,
    ShortestPathKind,
)

def get_shortest_path_directed(
    G: nx.DiGraph,
    source: int,
    nodes: NodeList,
) -> dict[int, NodeList]:
    if not G.is_directed():
        raise ValueError('Graph must be directed.')
    paths = {}
    for n in nodes:
        path = nx.shortest_path(G, source, n)
        paths[n] = {
            "length": len(path), 
            "path": path
        }
    return paths

def get_shortest_path_reversed(
    G: nx.DiGraph,
    source: int,
    nodes: NodeList,
) -> dict[int, NodeList]:
    if not G.is_directed():
        raise ValueError('Graph must be directed.')
    G = G.reverse()
    paths = {}
    for n in nodes:
        path = nx.shortest_path(G, source, n)
        paths[n] = {
            "length": len(path), 
            "path": path
        }
    return paths

def get_shortest_path_undirected(
    G: nx.Graph,
    source: int,
    nodes: NodeList,
) -> dict[int, NodeList]:
    G = G.to_undirected()
    paths = {}
    for n in nodes:
        path = nx.shortest_path(G, source, n)
        paths[n] = {
            "length": len(path), 
            "path": path
        }
    return paths

distance_functions = {
    'directed': get_shortest_path_directed,
    'reversed': get_shortest_path_reversed,
    'undirected': get_shortest_path_undirected,
}

def get_shortest_path(
    G: nx.Graph,
    source: int,
    nodes: NodeList,
    kind: ShortestPathKind,
):
    if kind not in distance_functions:
        raise ValueError(f'Invalid distance kind: {kind}')
    return distance_functions[kind](G, source, nodes)
