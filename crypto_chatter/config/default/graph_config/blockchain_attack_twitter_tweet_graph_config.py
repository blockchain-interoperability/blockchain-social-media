from crypto_chatter.config import (
    DATA_DIR,
    CryptoChatterGraphConfig,
)

index_name = 'blockchain-interoperability-attacks'

class BlockchainAttackTwitterTweetGraphConfig(CryptoChatterGraphConfig):
    graph_type = 'tweet'
    graph_dir = DATA_DIR / f'twitter/{index_name}/tweet-graph'
    node_id_col ='id'

    def __post_init__(self):
        self.graph_dir.mkdir(exist_ok=True,parents=True)
