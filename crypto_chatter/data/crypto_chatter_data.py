from typing_extensions import Self
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils.types import (
    Sentiment,
    IdList,
    TextList,
)

from .load_snapshots import load_snapshots
from .embeddings import get_sbert_embeddings
from .sentiment import get_roberta_sentiments
from .tfidf import fit_tfidf, get_tfidf

class CryptoChatterData:
    data_config: CryptoChatterDataConfig
    columns: list[str]
    available_columns: list[str] 
    df: pd.DataFrame|None = None
    tfidf: TfidfVectorizer | None = None
    tfidf_settings: str = ""
    cache_dir: Path | None = None
    lite_mode: bool = False
    ids: np.ndarray

    def __init__(
        self,
        data_config: CryptoChatterDataConfig,
        cols_to_load: list[str] | None = None,
        df: pd.DataFrame | None = None,
    ) -> None:
        if df is None:
            # if df is not provided, we are using cached mode. 
            self.cache_dir = data_config.data_dir / 'parsed'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.data_config = data_config
            if not self.is_built: self.build()
            if cols_to_load is not None: self.load(
                [self.data_config.id_col, self.data_config.text_col]+cols_to_load,
                refresh=True
            )
            # self.index = self.df.index.to_numpy()
        else:
            # if df is provided, we are using lite mode.
            self.lite_mode = True
            self.data_config = data_config
            self.df = df
            self.columns = df.columns.tolist()
            if data_config.text_col not in self.columns:
                raise ValueError(f'Text column [{data_config.text_col}] must be in columns for lite mode')

        self.ids = self.df[self.data_config.id_col].values
        self.df.index = self.ids

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
        # if refresh is True, we overwrite the previous columns
        new_cols = (
            cols_to_load 
            if refresh else
            [c for c in cols_to_load if c not in self.columns]
        )
        # drop duplicate columns
        new_cols = sorted(set(new_cols))
        new_df = pd.concat([
            pd.read_pickle(self.cache_dir / f'{c}.pkl')
            for c in new_cols
        ], axis=1)

        if self.df is None or refresh:
            # remove from memory if refreshing
            if self.df is not None: del self.df
            self.df = new_df
        else:
            self.df = pd.concat([self.df, new_df], axis=1)

        self.columns = self.df.columns.tolist()
        print(f'loaded {cols_to_load} in {int(time.time() - start)} seconds')
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, key: str|pd.Series|np.ndarray) -> pd.DataFrame|pd.Series:
        if isinstance(key, str) and key not in self.columns:
            self.load([key])
        return self.df[key]

    def text(
        self,
        ids: IdList|None = None,
    ) -> TextList:
        target_ids = (
            self.ids 
            if ids is None else 
            ids
        )
        return self.df[self.data_config.text_col][target_ids].values

    def sentiments(
        self,
        ids: IdList|None = None,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ) -> list[Sentiment]:
        target_ids = (
            self.ids 
            if ids is None else 
            ids
        )
        return get_roberta_sentiments(
            text=self.text(target_ids),
            data_config=self.data_config,
            ids=target_ids, 
            model_name=model_name
        )

    def embeddings(
        self,
        ids: IdList|None = None,
        model_name: str = "all-MiniLM-L12-v2"
    ) -> np.ndarray:
        target_ids = (
            self.ids 
            if ids is None else 
            ids
        )
        return get_sbert_embeddings(
            text=self.text(target_ids),
            data_config=self.data_config,
            ids=target_ids, 
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
    ) -> None:
        self.tfidf = fit_tfidf(
            self.df[self.data_config.text_col],
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
        texts: TextList,
    ) -> tuple[list[str], list[str]]:
        return get_tfidf(texts, self.tfidf)