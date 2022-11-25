import time
import pandas as pd
import numpy as np
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from collect_data import load_cache,load_pickles
from tokenizer import load_tokens
from embeddings import load_embeddings
from sentiment import load_sentiment


class TwitterDataset(Dataset):
    def __init__(
        self,
        timestamp_path,
        sentiment_path,
        whole_text_path = '',
        token_path = '',
        embedding_path = '',
    ):
        start = time.perf_counter()
        # data =
        timestamp = pd.to_datetime(load_pickles(timestamp_path),unit='ms')

        # 5%are missing timestamp... need to filter out
        
        # just empty list if we don't use it
        tokens = [None]* len(timestamp)
        if token_path:
            tokens = load_tokens(token_path)
        
        embeddings = [None]* len(timestamp)
        if embedding_path:
            embeddings = load_embeddings(embedding_path)

        whole_text = [None]* len(timestamp)
        if whole_text_path:
            whole_text = load_pickles(whole_text_path)
        
        sentiment_label,sentiment_score = load_sentiment(sentiment_path)
        print(f'loaded dataset. took {time.perf_counter()-start} ms')

        # now we only save the ones we use. discard the rest to save memory 
        # self.timestamp = timestamp
        self.whole_text = whole_text
        self.tokens = tokens
        self.embedding = embeddings
        self.sentiment_label = sentiment_label
        self.sentiment_score = sentiment_score

        sorted_idx = np.argsort(timestamp)

        missing_time = pd.isna(timestamp).nonzero()[0]

        # self.sorted_idx = list(filter(lambda i: not i in missing_time,sorted_idx))
        self.sorted_idx = sorted_idx[~np.in1d(sorted_idx,missing_time)]

        self.timestamp = timestamp.astype(int)

        del tokens,embeddings,sentiment_label,sentiment_score

    def __getitem__(self,idx):
        idx = self.sorted_idx[idx]
        return {
            'original_index': idx,
            # 'timestamp': self.timestamp[idx],
            # 'whole_text': self.whole_text[idx],
            # 'tokens': self.tokens[idx],
            'embedding': self.embedding[idx],
            'sentiment_label': self.sentiment_label[idx],
            # 'sentiment_score': self.sentiment_score[idx],
        }
        
    def __len__(self):
        return len(self.sorted_idx)

    def get_range(self,start_time,end_time):
        start = np.searchsorted(
            self.timestamp[self.sorted_idx],
            pd.to_datetime(start_time).value
        )

        end = np.searchsorted(
            self.timestamp[self.sorted_idx],
            pd.to_datetime(end_time).value
        )-1


        return start,end




# def crete_datasets(


# ):
