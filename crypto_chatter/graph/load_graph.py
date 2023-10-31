from typing import Literal

from crypto_chatter.config import CryptoChatterDataConfig

from .crypto_twitter_tweet_graph import CryptoTwitterTweetGraph
from .crypto_graph import CryptoGraph

def load_graph(
    data_config: CryptoChatterDataConfig,
) -> CryptoGraph:
    if (
        data_config.data_source == 'twitter' 
        and data_config.graph_type == 'tweet'
    ):
        graph = CryptoTwitterTweetGraph(data_config)
    else:
        raise NotImplementedError('Other data sources are not implemented')

    return graph
