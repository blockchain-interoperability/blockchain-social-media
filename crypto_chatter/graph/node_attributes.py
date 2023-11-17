import typing
import pandas as pd
import numpy as np
from functools import partial

from crypto_chatter.data import CryptoChatterData

from crypto_chatter.utils.types import (
    NodeList,
    NodeAttributeKind,
    NodeAttributeList,
    NodeAttributeDict,
    NodeToIndexMapping,
    SentimentKind,
)

class NodeAttribute:
    name: NodeAttributeKind
    values: NodeAttributeDict

    def __init__(
        self,
        name: NodeAttributeKind,
        nodes: NodeList,
        values: NodeAttributeList,
    ) -> None:
        self.name = name
        self.values = dict(zip(nodes, values))

def get_node_text(
    data: CryptoChatterData,
    nodes: NodeList,
) -> NodeAttribute:
    values = data.text(nodes)
    return NodeAttribute('text', nodes, values)

def get_node_sentiment(
    data: CryptoChatterData,
    nodes: NodeList,
    kind: SentimentKind,
) -> NodeAttribute:
    if not kind in typing.get_args(SentimentKind):
        raise ValueError(f'Unknown sentiment kind: {kind}')
    values = [s[kind] for s in data.sentiments(nodes)]
    return NodeAttribute('sentiment', nodes, values)

node_attr_functions = {
    'text': get_node_text,
    'sentiment_positive': partial(get_node_sentiment, kind='positive'),
    'sentiment_negative': partial(get_node_sentiment, kind='negative'),
    'sentiment_neutral': partial(get_node_sentiment, kind='neutral'),
}

def get_node_attribute(
    data: CryptoChatterData,
    nodes: NodeList,
    kind: NodeAttributeKind,
) -> NodeAttribute:
    if kind in node_attr_functions:
        return node_attr_functions[kind](
            data=data,
            nodes=nodes,
        )
    else:
        raise ValueError(f'Unknown node attribute kind: {kind}')

def get_node_attributes(
    data: CryptoChatterData,
    nodes: NodeList,
    kinds: list[NodeAttributeKind],
) -> list[NodeAttribute]:
    valid_nodes = [n for n in nodes if n in data.ids]
    return [
        get_node_attribute(
            data=data,
            nodes=valid_nodes,
            kind=k,
        )
        for k in kinds
    ]
