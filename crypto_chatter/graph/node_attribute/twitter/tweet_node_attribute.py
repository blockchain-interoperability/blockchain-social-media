import typing
import numpy as np
from functools import partial

from crypto_chatter.data import CryptoChatterData

from crypto_chatter.utils.types import (
    NodeList,
    TweetGraphNodeAttributeKind,
    NodeAttribute,
    TweetAttributeKind,
    AttributeValues,
    SentimentKind,
)

def get_node_sentiment(
    nodes: NodeList,
    data: CryptoChatterData,
    kind: SentimentKind,
) -> AttributeValues:
    if not kind in typing.get_args(SentimentKind):
        raise ValueError(f"Unknown sentiment kind: {kind}")
    print(f'trying to get sentiment {kind}')
    values = [
        s[kind] 
        for s in data.get("sentiment", nodes)
    ]
    return values

def get_node_attr(
    nodes: NodeList,
    data: CryptoChatterData,
    attr_name: TweetAttributeKind,
):
    if attr_name not in data.columns:
        data.load([attr_name])
    print(f'trying to get node attr {attr_name}')
    values = data.get(attr_name, nodes)
    return values

node_attr_functions = {
    "sentiment_positive": partial(get_node_sentiment, kind="positive"),
    "sentiment_negative": partial(get_node_sentiment, kind="negative"),
    "sentiment_neutral": partial(get_node_sentiment, kind="neutral"),
    "sentiment_composite": partial(get_node_sentiment, kind="composite"),
    "text": partial(get_node_attr, attr_name="text"),
    "retweet_count": partial(get_node_attr, attr_name="retweet_count"),
    "favorite_count": partial(get_node_attr, attr_name="favorite_count"),
    "quote_count": partial(get_node_attr, attr_name="quote_count"),
    "reply_count": partial(get_node_attr, attr_name="reply_count")
}

def get_tweet_node_attribute(
    nodes: NodeList,
    data: CryptoChatterData,
    kind: TweetGraphNodeAttributeKind,
) -> NodeAttribute:
    if kind in node_attr_functions:
        values = node_attr_functions[kind](
            nodes=nodes,
            data=data,
        )
        return dict(zip(nodes, values))
    else:
        raise ValueError(f"Unknown node attribute kind: {kind}")
