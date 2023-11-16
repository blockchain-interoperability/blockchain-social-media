from typing import Literal

from crypto_chatter.config import CryptoChatterDataConfig

from .blockchain_attack_twitter_tweet_graph_config import BlockchainAttackTwitterTweetGraphConfig

def load_default_graph_config(
    dataset:str, 
    graph_type: Literal['tweet','user'],
) -> CryptoChatterDataConfig:
    if (
        dataset == 'twitter:blockchain-interoperability-attacks' 
        and graph_type == 'tweet'
    ):
        data_config = BlockchainAttackTwitterTweetGraphConfig
    else:
        raise Exception('Unknown data source and graph combination')
    return data_config
