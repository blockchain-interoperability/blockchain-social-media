import networkx as nx

from crypto_chatter.utils.types import (
    NodeList, 
    CommunityKind,
)

def get_louvain_communities(
    G,
    random_seed: int,
) -> list[NodeList]:
    coms = nx.community.louvain_communities(
        G,
        seed = random_seed
    )
    coms = [list(c) for c in sorted(coms, key=len, reverse=True)]
    return coms

community_functions = {
    'louvain': get_louvain_communities,
}

def get_communities(
    G,
    community_kind: CommunityKind,
    random_seed,
) -> list[NodeList]:
    if community_kind not in community_functions:
        raise ValueError(f'Unknown community kind: {community_kind}')
    return community_functions[community_kind](G, random_seed)
