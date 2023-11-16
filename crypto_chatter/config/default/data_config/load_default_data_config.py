from crypto_chatter.config import CryptoChatterDataConfig

from .blockchain_attack_twitter_data_config import BlockchainAttackTwitterDataConfig

def load_default_data_config(
    dataset:str, 
) -> CryptoChatterDataConfig:
    if dataset == 'twitter:blockchain-interoperability-attacks':
        data_config = BlockchainAttackTwitterDataConfig
    else:
        raise Exception('Unknown data source')
    return data_config
