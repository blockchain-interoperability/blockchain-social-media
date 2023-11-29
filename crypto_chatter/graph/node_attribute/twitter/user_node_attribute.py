import typing
import numpy as np
from functools import partial
from rich.progress import Progress

from crypto_chatter.data import CryptoChatterData

from crypto_chatter.utils import aggr_funcs
from crypto_chatter.utils.types import (
    NodeList,
    NodeToIdMap,
    UserAttributeKind,
    AggrFunction,
    UserGraphNodeAttributeKind,
    NodeAttribute,
    AttributeValues,
    SentimentKind,
)

def get_node_avg_sentiment(
    nodes: NodeList,
    node_to_ids: NodeToIdMap,
    data: CryptoChatterData,
    kind: SentimentKind,
    progress: Progress | None = None,
) -> AttributeValues:
    if not kind in typing.get_args(SentimentKind):
        raise ValueError(f'Unknown sentiment kind: {kind}')

    if progress is not None:
        task = progress.add_task(
            f"getting {kind} sentiment",
            total=len(nodes),
        )
    values = []
    for n in nodes:
        values += [np.mean([
            s[kind] 
            for s in data.get('sentiment', node_to_ids[n])
        ])]
        if progress is not None:
            progress.advance(task)
    if progress is not None:
        progress.remove_task(task)

    return values

def get_user_attr(
    nodes: NodeList,
    node_to_ids: NodeToIdMap,
    data: CryptoChatterData,
    attr_name: UserAttributeKind,
    aggr_func: AggrFunction,
    progress: Progress | None = None,
):
    if not attr_name in data.columns:
        data.load([attr_name])

    if progress is not None:
        task = progress.add_task(
            f"getting {attr_name}",
            total=len(nodes),
        )
    values = []
    for n in nodes:
        values += [aggr_funcs[aggr_func](data.get(attr_name, node_to_ids[n]))]
        if progress is not None:
            progress.advance(task)
    if progress is not None:
        progress.remove_task(task)

    return values

node_attr_functions = {
    'avg_sentiment_positive': partial(get_node_avg_sentiment, kind='positive'),
    'avg_sentiment_negative': partial(get_node_avg_sentiment, kind='negative'),
    'avg_sentiment_neutral': partial(get_node_avg_sentiment, kind='neutral'),
    'followers_count': partial(get_user_attr, attr_name='user.followers_count', aggr_func="first"),
    'friends_count': partial(get_user_attr, attr_name='user.friends_count', aggr_func="first"),
    'avg_retweet_count': partial(get_user_attr, attr_name='retweet_count', aggr_func="mean"),
    'avg_favorite_count': partial(get_user_attr, attr_name='favorite_count', aggr_func="mean"),
    'avg_reply_count': partial(get_user_attr, attr_name='reply_count', aggr_func="mean"),
    'avg_quote_count': partial(get_user_attr, attr_name='quote_count', aggr_func="mean"),
    'total_retweet_count': partial(get_user_attr, attr_name='retweet_count', aggr_func="sum"),
    'total_favorite_count': partial(get_user_attr, attr_name='favorite_count', aggr_func="sum"),
    'total_reply_count': partial(get_user_attr, attr_name='reply_count', aggr_func="sum"),
    'total_quote_count': partial(get_user_attr, attr_name='quote_count', aggr_func="sum"),
}

def get_user_node_attribute(
    nodes: NodeList,
    node_to_ids: NodeToIdMap,
    data: CryptoChatterData,
    kind: UserGraphNodeAttributeKind,
    progress: Progress | None = None,
) -> NodeAttribute:
    if kind in node_attr_functions:
        values = node_attr_functions[kind](
            nodes=nodes,
            node_to_ids=node_to_ids,
            data=data,
            progress=progress,
        )
        return dict(zip(nodes, values))
    else:
        raise ValueError(f'Unknown node attribute kind: {kind}')
