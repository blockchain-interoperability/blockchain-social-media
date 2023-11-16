import pandas as pd

from crypto_chatter.config import CryptoChatterDataConfig

from .load_raw_data import load_raw_data
from .embeddings import generate_sbert_embeddings, get_sbert_embedding
from .sentiment import generate_roberta_sentiment, get_roberta_sentiment

class CryptoChatterData:
    data_config: CryptoChatterDataConfig
    df: pd.DataFrame

    def __init__(
        self,
        data_config: CryptoChatterDataConfig,
    ) -> None:
        self.data_config = data_config
        self.build()

    def build(self) -> None:
        self.df = load_raw_data(self.data_config)
        # generate_sbert_embeddings(self.df[self.data_config.text_col], self.data_config)
        generate_roberta_sentiment(self.df[self.data_config.text_col], self.data_config)

    def sentiment(
        self,
        row_index: int,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ):
        get_roberta_sentiment(row_index, self.data_config, model_name)

    def embedding(
        self,
        row_index: int,
        model_name: str = "all-MiniLM-L12-v2"
    ):
        get_sbert_embedding(row_index, self.data_config, model_name)
