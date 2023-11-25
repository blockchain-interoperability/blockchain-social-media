from crypto_chatter.data import CryptoChatterData
from crypto_chatter.config import CryptoChatterDataConfig

dataset = "twitter:blockchain-interoperability-attacks"
graph_type = "tweet"

data_config = CryptoChatterDataConfig(dataset)
data = CryptoChatterData(
    data_config=data_config,
    columns=["hashtags"],
)

