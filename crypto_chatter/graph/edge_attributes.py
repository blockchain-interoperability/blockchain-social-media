from typing import Literal
import numpy as np

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.config import CryptoChatterGraphConfig
from crypto_chatter.utils.types import (
    EdgeList,
    EdgeAttributeKind,
    EdgeAttributeList,
    EdgeAttributeDict,
) 

class EdgeAttribute:
    name: str
    edges: EdgeList
    values: EdgeAttributeDict

    def __init__(
        self,
        name: str,
        edges: EdgeList,
        values: EdgeAttributeList,
    ) -> None:
        self.name = name
        self.values = dict(zip(edges, values))

def get_edge_emb_cosine_similarity(
    edges: EdgeList,
    data: CryptoChatterData,
) -> EdgeAttribute:
    nodes = list(set([n for edge in edges for n in edge]))
    embeddings = dict(zip(
        nodes,
        data.embeddings(nodes),
    ))

    values = [
        (
            (embeddings[node_a] @ embeddings[node_b].T) 
            / (np.linalg.norm(embeddings[node_a]) * np.linalg.norm(embeddings[node_b]))
        )
        for node_a, node_b in edges
    ]

    return EdgeAttribute('emb_cosine_sim', edges, values)

edge_attr_functions = {
    'emb_cosine_sim': get_edge_emb_cosine_similarity,
}

def get_edge_attribute(
    data: CryptoChatterData,
    edges: EdgeList,
    attribute_name: EdgeAttributeKind,
) -> EdgeAttribute:
    if attribute_name == 'embedding_cosine_similarity':
        return compute_cosine_similarity(
            edges=edges,
            data=data
        )

def get_edge_attributes(
    data: CryptoChatterData,
    edges: EdgeList,
    attribute_names: list[EdgeAttributeKind],
) -> list[EdgeAttribute]:
    valid_edges = [e for e in edges if e[0] in data.ids and e[1] in data.ids]
    return [
        get_edge_attribute(
            data=data, 
            edges=valid_edges,
            attribute_name=a,
            graph_config=graph_config,
        )
        for a in attribute_names
    ]

