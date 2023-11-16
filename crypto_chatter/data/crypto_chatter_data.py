import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time

from crypto_chatter.config import CryptoChatterDataConfig

from .load_snapshots import load_snapshots
from .embeddings import generate_sbert_embeddings, get_sbert_embedding
from .sentiment import generate_roberta_sentiment, get_roberta_sentiment
from .tfidf import fit_tfidf, get_tfidf

class CryptoChatterData:
    data_config: CryptoChatterDataConfig
    columns: list[str]
    available_columns: list[str] 
    text: pd.Series
    df: pd.DataFrame|None = None
    tfidf: TfidfVectorizer | None = None
    tfidf_settings: str = ""
    cache_dir: Path | None = None
    lite_mode: bool = False
    index: np.ndarray

    def __init__(
        self,
        data_config: CryptoChatterDataConfig,
        cols_to_load: list[str] | None = None,
        df: pd.DataFrame | None = None,
        index: list[int] | None = None,
    ) -> None:
        if df is None:
            # if df is not provided, we are using cached mode. 
            self.cache_dir = data_config.data_dir / 'parsed'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.data_config = data_config
            if not self.is_built: self.build()
            if cols_to_load is not None: self.load(cols_to_load, refresh=True)
            self.text = self[data_config.text_col]
            self.index = self.df.index.to_numpy()
        else:
            # if df is provided, we are using lite mode.
            self.lite_mode = True
            self.data_config = data_config
            self.df = df
            self.columns = df.columns.tolist()
            self.text = self[data_config.text_col]
            # Just in case the indices get messed up... we are treating them like PKs
            if index is not None:
                if isinstance(index, np.ndarray):
                    self.index = index
                elif isinstance(index, list):
                    self.index = np.array(index)
                else:
                    raise ValueError('index must be a list or np.ndarray')
            else:
                self.index = self.df.index.to_numpy()

    @property
    def is_built(self):
        return (self.cache_dir/'completed.txt').is_file() or self.lite_mode

    def build(self) -> None:
        # Only happens on the first time. Populates the columns into pickles inside the cache folder
        if self.is_built or self.lite_mode: return
        print('Building CyrptoChatterData..')
        start = time.time()
        df = load_snapshots(self.data_config)
        for c in df.columns:
            df[c].to_pickle(self.cache_dir / f'{c}.pkl')
        (self.cache_dir/'completed.txt').touch()
        self.available_columns = df.columns.tolist()
        del df
        print(f'Built CryptoChatterData in {int(time.time() - start)} seconds')

    def load(
        self,
        cols_to_load:list[str],
        refresh: bool = False,
    ) -> None:
        # loads columns and ignores ones already loaded
        print(f'loading {cols_to_load}..')
        if self.lite_mode: return

        start = time.time()
        new_cols = (
            cols_to_load 
            if refresh else
            [c for c in cols_to_load if c not in self.columns]
        )
        new_df = pd.concat([
            pd.read_pickle(self.cache_dir / f'{c}.pkl')
            for c in new_cols
        ], axis=1)
        if self.df is None or refresh:
            if self.df is not None: del self.df
            self.df = new_df
            self.columns = new_cols
        else:
            self.df = pd.concat([self.df, new_df], axis=1)
            self.columns += new_cols
        print(f'loaded {cols_to_load} in {int(time.time() - start)} seconds')
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, key: str|pd.Series|np.ndarray) -> pd.DataFrame|pd.Series:
        if isinstance(key, str) and key not in self.columns:
            self.load([key])
        return self.df[key]

    def generate_embeddings(self) -> None:
        generate_sbert_embeddings(
            text=self.text,
            data_config=self.data_config,
            indices = self.index,
        )

    def generate_sentiment(self) -> None:
        generate_roberta_sentiment(
            text=self.text,
            data_config=self.data_config,
            indices = self.index,
        )

    def sentiment(
        self,
        index: int,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ):
        return get_roberta_sentiment(
            row_index=self.index[index], 
            data_config=self.data_config,
            text=self.text.values[index],
            model_name=model_name
        )

    def embedding(
        self,
        index: int,
        model_name: str = "all-MiniLM-L12-v2"
    ):
        return get_sbert_embedding(
            row_index=self.index[index], 
            data_config=self.data_config,
            text=self.text.values[index],
            model_name=model_name
        )

    def fit_tfidf(
        self, 
        random_seed:int = 0,
        random_size:int = 1000000,
        ngram_range:tuple[int,int] = (1, 1),
        max_df:float|int = 1.0,
        min_df:float|int = 1,
        max_features:int = 10000,
    ):
        self.tfidf = fit_tfidf(
            self.text.values,
            self.data_config,
            random_seed = random_seed,
            random_size = random_size,
            ngram_range = ngram_range,
            max_df = max_df,
            min_df = min_df,
            max_features = max_features,
        )

    def get_tfidf(
        self,
        text: list[str]|np.ndarray|pd.Series,
    ):
        return get_tfidf(text, self.tfidf)
