import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.progress import Progress
import time

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils.types import (
    IdList,
    TextList,
)

from .load_snapshots import load_snapshots
from .embeddings import get_sbert_embeddings
from .sentiment import get_roberta_sentiments, Sentiment
from .tfidf import fit_tfidf, get_tfidf, TfidfConfig

class CryptoChatterData:
    data_config: CryptoChatterDataConfig
    columns: list[str]
    available_columns: list[str] 
    tfidf: TfidfVectorizer | None = None
    cache_dir: Path | None = None
    ids: np.ndarray
    progress: Progress|None
    use_progress: bool = False
    tfidf_config: TfidfConfig | None

    def __init__(
        self,
        data_config: CryptoChatterDataConfig,
        columns: list[str] = [],
        progress: Progress|None = None,
    ) -> None:
        # if df is not provided, we are using cached mode. 
        self.cache_dir = data_config.data_dir / 'parsed'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_config = data_config
        self.progress = progress
        self.use_progress = progress is not None

        if not self.is_built:
            self.build()
        else:
            self.available_columns = [f.stem for f in self.cache_dir.glob('*.pkl')]
        self.load(
            [self.data_config.id_col, self.data_config.text_col]+columns,
            refresh=True
        )
        self.reset_ids()

    @property
    def is_built(self):
        return (self.cache_dir/'completed.txt').is_file()

    def reset_ids(self):
        self.ids = self.df[self.data_config.id_col].values
        self.df.index = self.ids

    def build(self) -> None:
        # Only happens on the first time. Populates the columns into pickles inside the cache folder
        if self.is_built: return
        print('Building CyrptoChatterData..')
        start = time.time()
        df = load_snapshots(
            data_config=self.data_config,
            progress=self.progress,
        )
        df.index = df[self.data_config.id_col].values

        if self.use_progress:
            save_task = self.progress.add_task(
                description='saving columns..',
                total=len(df.columns),
            )

        for c in df.columns:
            df[c].to_pickle(self.cache_dir / f'{c}.pkl')
            if self.use_progress:
                self.progress.advance(save_task)
        if self.use_progress:
            self.progress.remove_task(save_task)

        (self.cache_dir/'completed.txt').touch()
        self.available_columns = df.columns.tolist()
        del df
        print(f'Built CryptoChatterData in {time.time() - start:.2f} seconds')

    def has_ids( 
        self,
        ids: IdList
    ) -> np.ndarray:
        if not isinstance(ids, np.ndarray): 
            ids = np.array(ids)
        mask = np.isin(ids, self.ids, assume_unique=True)
        return mask
    
    def drop(
        self,
        columns: list[str],
    ) -> None:
        if any(c not in self.columns for c in columns):
            raise ValueError(f'Unknown columns: {columns}')
        for c in columns:
            del self.df[c]
        self.columns = [c for c in self.columns if c not in columns]

    def load(
        self,
        columns:list[str],
        refresh: bool = False,
    ) -> None:
        if not columns: return
        if any(c not in self.available_columns for c in columns):
            raise ValueError(f'Unknown columns: {columns}')
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
            print(f'refreshed with {columns} in {time.time() - start:.2f} seconds')

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
                print(f'loaded {new_cols} in {time.time() - start:.2f} seconds')

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
        elif col == 'clean_text':
            return 

        elif col in self.columns:
            return self.df[col].loc[target_ids].values
        else:
            raise ValueError(f'Unknown column: {col}')

    def fit_tfidf(
        self, 
        random_seed:int = 0,
        random_size:int = int(1e6),
        ngram_range:tuple[int,int] = (1, 1),
        max_df:float|int = 1.0,
        min_df:float|int = 1,
        max_features:int = int(1e4),
    ) -> None:
        self.tfidf_config = TfidfConfig(
            random_seed=random_seed,
            random_size=random_size,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
        )
        self.tfidf = fit_tfidf(
            text=self.get('text'),
            cache_dir = self.data_config.data_dir,
            config=self.tfidf_config,
        )

    def get_tfidf(
        self,
        texts: TextList,
    ) -> dict[str,float]:
        if self.tfidf is None:
            raise ValueError("tfidf is not fitted")
        return get_tfidf(texts, self.tfidf)
