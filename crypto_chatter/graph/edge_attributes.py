import numpy as np
from rich.progress import Progress

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.utils.types import (
    EdgeList,
    EdgeAttributeKind,
    EdgeAttribute,
    AttributeValues,
) 

def get_edge_emb_cosine_similarity(
    data: CryptoChatterData,
    edges: EdgeList,
    # progress: Progress | None = None,
) -> AttributeValues:
    nodes = list(set([n for edge in edges for n in edge]))
    embeddings = dict(zip(
        nodes,
        data.get('embedding',nodes),
    ))

    # if progress is not None:
    #     task = progress.add_task('Calculating edge embeddings', total=len(edges))

    values = []
    for node_a, node_b in edges:
        values += [float(
            (embeddings[node_a] @ embeddings[node_b].T) 
            / (np.linalg.norm(embeddings[node_a]) * np.linalg.norm(embeddings[node_b]))
        )]

    #     if progress is not None:
    #         progress.advance(task)

    # if progress is not None:
    #     progress.remove_task(task)

    return values

edge_attr_functions = {
    'emb_cosine_sim': get_edge_emb_cosine_similarity,
}

def get_edge_attribute(
    edges: EdgeList,
    data: CryptoChatterData,
    kind: EdgeAttributeKind,
    # progress: Progress | None = None,
) -> EdgeAttribute:
    if kind in edge_attr_functions:
        values = edge_attr_functions[kind](
            edges=edges,
            data=data,
            # progress=progress,
        )
        return dict(zip(edges, values))
    else:
        raise ValueError(f'Unknown node attribute kind: {kind}')
