from crypto_chatter.data import CryptoChatterData
from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import progress_bar

dataset = "twitter:blockchain-interoperability-attacks"
graph_type = "tweet"

data_config = CryptoChatterDataConfig(dataset)
with progress_bar() as progress:
    data = CryptoChatterData(
        data_config=data_config,
        # columns=["hashtags"],
        progress=progress,
    )

