import numpy as np

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.utils.types import (
    EdgeList,
    TweetGraphEdgeAttributeKind,
    EdgeAttribute,
    AttributeValues,
) 

def get_edge_emb_cosine_similarity(
    data: CryptoChatterData,
    edges: EdgeList,
) -> AttributeValues:
    nodes = list(set([n for edge in edges for n in edge]))
    embeddings = dict(zip(
        nodes,
        data.get('embedding',nodes),
    ))

    values = []
    for node_a, node_b in edges:
        values += [float(
            (embeddings[node_a] @ embeddings[node_b].T) 
            / (np.linalg.norm(embeddings[node_a]) * np.linalg.norm(embeddings[node_b]))
        )]

    return values

edge_attr_functions = {
    'emb_cosine_sim': get_edge_emb_cosine_similarity,
}

def get_tweet_edge_attribute(
    edges: EdgeList,
    data: CryptoChatterData,
    kind: TweetGraphEdgeAttributeKind,
) -> EdgeAttribute:
    if kind in edge_attr_functions:
        values = edge_attr_functions[kind](
            edges=edges,
            data=data,
            # progress=progress,
        )
        return dict(zip(map(tuple, edges), values))
    else:
        raise ValueError(f'Unknown node attribute kind: {kind}')
