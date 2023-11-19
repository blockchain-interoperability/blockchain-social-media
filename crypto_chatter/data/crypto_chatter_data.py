import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.progress import Progress
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
    progress: Progress|None
    use_progress: bool = False

    def __init__(
        self,
        data_config: CryptoChatterDataConfig,
        columns: list[str] = [],
        df: pd.DataFrame | None = None,
        progress: Progress|None = None,
    ) -> None:
        if df is None:
            # if df is not provided, we are using cached mode. 
            self.cache_dir = data_config.data_dir / 'parsed'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.data_config = data_config
            if not self.is_built: self.build()
            self.load(
                [self.data_config.id_col, self.data_config.text_col]+columns,
                refresh=True
            )
        else:
            # if df is provided, we are using lite mode.
            self.lite_mode = True
            self.data_config = data_config
            self.df = df
            self.columns = df.columns.tolist()
            if data_config.text_col not in self.columns:
                raise ValueError(f'Text column [{data_config.text_col}] must be in columns for lite mode')

        self.progress = progress
        self.use_progress = progress is not None

        self.reset_ids()

    @property
    def is_built(self):
        return (self.cache_dir/'completed.txt').is_file() or self.lite_mode

    def reset_ids(self):
        self.ids = self.df[self.data_config.id_col].values
        self.df.index = self.ids

    def build(self) -> None:
        # Only happens on the first time. Populates the columns into pickles inside the cache folder
        if self.is_built or self.lite_mode: return
        print('Building CyrptoChatterData..')
        start = time.time()
        df = load_snapshots(
            data_config=self.data_config,
            progress=self.progress,
        )
        df.index = df[self.data_config.id_col].values
        for c in df.columns:
            df[c].to_pickle(self.cache_dir / f'{c}.pkl')
        (self.cache_dir/'completed.txt').touch()
        self.available_columns = df.columns.tolist()
        del df
        print(f'Built CryptoChatterData in {int(time.time() - start)} seconds')

    def has_ids( 
        self,
        ids: IdList
    ) -> np.ndarray:
        if not isinstance(ids, np.ndarray): 
            ids = np.array(ids)
        mask = np.isin(ids, self.ids, assume_unique=True)
        return mask

    def load(
        self,
        columns:list[str],
        refresh: bool = False,
    ) -> None:
        if self.lite_mode or not columns: return

        columns = sorted(set(columns))
        print(f'loading {columns}..')

        # if refresh is True, we overwrite the previous columns
        if refresh:
            start = time.time()
            try: del self.df
            except AttributeError: pass

            if self.use_progress:
                progress_task = self.progress.add_task(
                    description='loading columns',
                    total=len(columns),
                )

            loaded_cols = []
            for c in columns:
                loaded_cols += [pd.read_pickle(self.cache_dir / f'{c}.pkl')]
                if self.use_progress:
                    self.progress.advance(progress_task)

            if self.use_progress:
                self.progress.remove_task(progress_task)

            self.df = pd.concat(loaded_cols, axis=1)
            print(f'refreshed with {columns} in {int(time.time() - start)} seconds')

        else:
            new_cols = (
                columns 
                if refresh else
                [c for c in columns if c not in self.columns]
            )
            # drop duplicate columns
            if new_cols:
                start = time.time()

                if self.use_progress:
                    progress_task = self.progress.add_task(
                        description='loading columns',
                        total=len(columns),
                    )

                loaded_cols = []
                for c in new_cols:
                    loaded_cols += [
                        pd.read_pickle(self.cache_dir / f'{c}.pkl')
                    ]

                    if self.use_progress:
                        self.progress.advance(progress_task)

                if self.use_progress:
                    self.progress.remove_task(progress_task)

                new_df = pd.concat(loaded_cols, axis=1)
                self.df = pd.concat([self.df, new_df], axis=1)
                print(f'loaded {new_cols} in {int(time.time() - start)} seconds')

        self.columns = self.df.columns.tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, key: str|pd.Series|np.ndarray) -> pd.DataFrame|pd.Series:
        if isinstance(key, str) and key not in self.columns:
            self.load([key])
        return self.df[key]

    def get(
        self,
        col: str,
        ids: IdList|None = None,
        **kwargs,
    ) -> TextList|list[Sentiment]|np.ndarray:
        target_ids = (
            self.ids 
            if ids is None else 
            ids
        )
        # commenting this check out b/c it takes way too long.
        # if any(i not in self.ids for i in target_ids):
        #     raise ValueError('Invalid ids provided')

        if col == 'text':
            return self.df[self.data_config.text_col].loc[target_ids].values
        elif col == 'sentiment':
            model_name = kwargs.get('model_name', "cardiffnlp/twitter-roberta-base-sentiment-latest")
            return get_roberta_sentiments(
                text=self.get('text',target_ids),
                data_config=self.data_config,
                ids=target_ids, 
                model_name=model_name,
                progress=self.progress,
            )
        elif col == 'embedding':
            model_name = kwargs.get('model_name', "all-MiniLM-L12-v2")
            return get_sbert_embeddings(
                text=self.get('text',target_ids),
                data_config=self.data_config,
                ids=target_ids, 
                model_name=model_name,
                progress=self.progress,
            )
        elif col in self.columns:
            return self.df[col].loc[target_ids].values

        else:
            raise ValueError(f'Unknown column: {col}')

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
            self.text(),
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
