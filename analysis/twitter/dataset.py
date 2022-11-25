import time
import pandas as pd
import numpy as np
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from collect_data import load_cache
from tokenizer import load_tokens
from embeddings import load_embeddings
from sentiment import load_sentiment

@dataclass
class TwitterData:
    timetamp: pd.Timestamp
    whole_text: str
    tokens:list
    embedding: torch.Tensor
    sentiment_label: list
    sentiment_score: list

class TwitterDataset(Dataset):
    def __init__(
        self,
        snapshot_path,
        sentiment_path,
        token_path = '',
        embedding_path = '',
    ):
        start = time.perf_counter()
        data = load_cache(snapshot_path)
        
        if token_path:
            tokens = load_tokens(token_path)
        else:
            # just empty list if we don't use it
            tokens = [None]* len(data)
        
        if embedding_path:
            embeddings = load_embeddings(embedding_path)
        else:
            embeddings = [None]* len(data)
        
        sentiment_label,sentiment_score = load_sentiment(sentiment_path)
        print(f'loaded dataset. took {time.perf_counter()-start} ms')

        start = time.perf_counter()
        self.timestamp = pd.to_datetime(data['timestamp_ms'],unit='ms')
        print(f'timestamp conversion complete. took {time.perf_counter()-start} ms')
        
        # now we only save the ones we use. discard the rest to save memory 
        # self.timestamp = timestamp
        self.whole_text = data.whole_text.values
        self.tokens = tokens
        self.embedding = embeddings
        self.sentiment_label = sentiment_label
        self.sentiment_score = sentiment_score

        self.sorted_idx = np.argsort(self.timestamp)


        del data,tokens,embeddings,sentiment_label,sentiment_score

    def __getitem__(self,idx):
        idx = self.sorted_idx[idx]
        return TwitterData(
            self.timestamp[idx],
            self.whole_text[idx],
            self.tokens[idx],
            self.embedding[idx],
            self.sentiment_label[idx],
            self.sentiment_score[idx],
        )
        
    def __len__(self):
        return len(self.sorted_idx)


# def crete_datasets(


# ):
