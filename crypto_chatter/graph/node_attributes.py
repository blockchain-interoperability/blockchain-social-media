import typing
from functools import partial

from crypto_chatter.data import CryptoChatterData

from crypto_chatter.utils.types import (
    NodeList,
    NodeAttributeKind,
    NodeAttribute,
    AttributeValues,
    SentimentKind,
)

def get_node_text(
    nodes: NodeList,
    data: CryptoChatterData,
) -> AttributeValues:
    values = data.get('text', nodes)
    return values

def get_node_sentiment(
    nodes: NodeList,
    data: CryptoChatterData,
    kind: SentimentKind,
) -> AttributeValues:
    if not kind in typing.get_args(SentimentKind):
        raise ValueError(f'Unknown sentiment kind: {kind}')
    values = [s[kind] for s in data.get('sentiment',nodes)]
    return values

node_attr_functions = {
    'text': get_node_text,
    'sentiment_positive': partial(get_node_sentiment, kind='positive'),
    'sentiment_negative': partial(get_node_sentiment, kind='negative'),
    'sentiment_neutral': partial(get_node_sentiment, kind='neutral'),
}

def get_node_attribute(
    nodes: NodeList,
    data: CryptoChatterData,
    kind: NodeAttributeKind,
) -> NodeAttribute:
    if kind in node_attr_functions:
        values = node_attr_functions[kind](
            nodes=nodes,
            data=data,
        )
        return dict(zip(nodes, values))
    else:
        raise ValueError(f'Unknown node attribute kind: {kind}')
