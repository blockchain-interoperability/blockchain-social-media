from .blockchain_attack_twitter_tweet_graph import BlockchainAttackTwitterTweetGraphConfig


def load_default_data_config(dataset:str, graph_type:str) -> CryptoChatterDataConfig:
    if dataset == 'twitter:blockchain-interoperability-attacks' and graph_type == 'tweet':
        data_config = BlockchainAttackTwitterTweetGraphConfig()
    else:
        raise Exception('Unknown data source and graph combination')
    return data_config
