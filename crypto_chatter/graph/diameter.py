import networkx as nx

from crypto_chatter.utils.types import (
    DiameterKind,
)

def get_diameter_directed(
    G: nx.DiGraph
):
    if not G.is_directed():
        raise ValueError('Graph must be directed')
    diameter = -1
    if nx.is_strongly_connected(G):
        diameter = nx.diameter(G)

    return diameter

def get_diameter_undirected(
    G: nx.Graph
):
    undir = G.to_undirected()
    diameter = -1
    if nx.is_connected(undir):
        diameter = nx.diameter(G)
    return diameter

diameter_functions = {
    'directed': get_diameter_directed,
    'undirected': get_diameter_undirected,
}


def get_diameter(
    G: nx.Graph,
    kind: DiameterKind
):
    return diameter_functions[kind](G)
