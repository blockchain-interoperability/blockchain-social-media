import networkx as nx

from crypto_chatter.utils.types import (
    ReachableKind,
    NodeList,
) 

def get_reachable_directed(
    G: nx.DiGraph,
    node: int,
) -> NodeList:
    if not G.is_directed():
        raise ValueError('Graph must be directed.')
    return list(nx.dfs_preorder_nodes(G, node))

def get_reachable_directed_reversed(
    G: nx.DiGraph,
    node: int,
) -> NodeList:
    if not G.is_directed():
        raise ValueError('Graph must be directed.')
    G = G.reverse()
    return list(nx.dfs_preorder_nodes(G, node))

def get_reachable_undirected(
    G: nx.Graph,
    node: int,
) -> NodeList:
    G = G.to_undirected()
    return list(nx.dfs_preorder_nodes(G, node))

reachable_functions = {
    'undirected': get_reachable_undirected,
    'directed': get_reachable_directed,
    'reversed': get_reachable_directed,
}

def get_reachable(
    G: nx.Graph,
    node: int,
    kind: ReachableKind,
):
    if kind not in reachable_functions:
        raise ValueError(f'Unknown reachable kind: {kind}')
    return reachable_functions[kind](
        G=G,
        node=node
    )
