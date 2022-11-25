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
    embedding: torch.Tensor
    sentiment_label: list
    sentiment_score: list

class TwitterDataset(Dataset):
    def __init__(self,snapshot_path,token_path,embedding_path,sentiment_path):
        start = time.perf_counter()
        data = load_cache(snapshot_path)
        tokens = load_tokens(token_path)
        embeddings = load_embeddings(embedding_path)
        sentiment_label,sentiment_score = load_sentiment(sentiment_path)
        print(f'loaded dataset. took {time.perf_counter()-start} ms')

        start = time.perf_counter()
        self.timestamp = pd.to_datetime(data['timestamp_ms'],unit='ms')
        print(f'timestamp conversion complete. took {time.perf_counter()-start} ms')
        self.whole_text = data.whole_text.copy()
        self.tokens = tokens
        self.embedding = embeddings
        self.sentiment_label = sentiment_label
        self.sentiment_score = sentiment_score

        self.sorted_idx = np.argsort(self.timestamp)

        del data

    def __get__(self,idx):
        return TwitterData(
            self.timestamp[self.sorted_idx[idx]],
            self.whole_text[self.sorted_idx[idx]],
            self.tokens[self.sorted_idx[idx]],
            self.embedding[self.sorted_idx[idx]],
            self.sentiment_label[self.sorted_idx[idx]],
            self.sentiment_score[self.sorted_idx[idx]],
        )
        

    def __len__(self):
        return len(self.sorted_idx)
