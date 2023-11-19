from crypto_chatter.data import CryptoChatterData
from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import progress_bar

progress = progress_bar()
progress.start()

top_n = 100
dataset = 'twitter:blockchain-interoperability-attacks'
graph_type = 'tweet'
data_config = CryptoChatterDataConfig(dataset)
data = CryptoChatterData(
    data_config,
    columns = ['hashtags'],
    progress = progress,
)

data.get('embedding')
progress.stop()
