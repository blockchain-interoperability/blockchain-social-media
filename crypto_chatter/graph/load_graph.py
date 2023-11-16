from typing import Literal

from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig

from .crypto_twitter_tweet_graph import CryptoChatterTwitterTweetGraph
from .crypto_chatter_graph import CryptoChatterGraph

def load_graph(
    data_config: CryptoChatterDataConfig,
    graph_config: CryptoChatterGraphConfig,
) -> CryptoChatterGraph:
    if (
        data_config.data_source == 'twitter' 
        and graph_config.graph_type == 'tweet'
    ):
        graph = CryptoChatterTwitterTweetGraph(data_config, graph_config)
    else:
        raise NotImplementedError('Other data sources are not implemented')

    return graph
