import time
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
from pathlib import Path
from tqdm.auto import tqdm


spam_patterns = [
    "I Wish I discovered this earlier. Uniswap is being exploited"
]

def is_spam(text):
    return any([s in text for s in spam_patterns])


def load_pickles(some_path):
    some_path = Path(some_path)
    result = []
    for f in tqdm(
        sorted(some_path.glob('*.pkl')),
        desc=f'loading from {some_path.name}',
        leave=False
    
    ):
        result += pickle.load(open(f,'rb'))
    # [(f) ]
    return result

class TwitterDataset(Dataset):
    def __init__(
        self,
        snapshot_path = '',
        embedder = '',
        embeddings_path = '',
        # timestamp_path,
        # spam_idx_path,
        # sentiment_path = '',
        # token_path = '',
        # embedding_path = '',
        # whole_text_path = '',
        # filter_spam = True,
    ):
        snapshot_path = Path(snapshot_path)
        embeddings_path = Path(embeddings_path)
        start = time.perf_counter()
        # data =
        timestamp = pd.to_datetime(pd.read_pickle(snapshot_path/'timestamp_ms'),unit='ms')

        # 5%are missing timestamp... need to filter out
        
        # just empty list if we don't use it
        sentiment_label,sentiment_score = [None]* len(timestamp),[None]* len(timestamp)
        
        # self.use_sentiment = False
        # if sentiment_path:
        #     sentiment_label,sentiment_score = load_sentiment(sentiment_path)
        #     self.use_sentiment=True

        # tokens = [None]* len(timestamp)
        # self.use_tokens = False
        # if token_path:
        #     tokens = load_tokens(token_path)
        #     self.use_tokens = True
        
        self.use_embeddings = False
        embeddings = [None]* len(timestamp)
        if embedding_path:
            embeddings = torch.load(embedding_path)
            self.use_embeddings = True

        whole_text = [None]* len(timestamp)
        self.use_whole_text = False
        if whole_text_path:
            whole_text = pd.read_pickle(snapshot_path/'whole_text')
            self.use_whole_text = True
        

        

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
        # if filter_spam:
        #     print('no spam here!')
        #     spam_idx = np.array(pickle.load(open(spam_idx_path,'rb')))
        #     # spam_tweet = np.array([i for i,t in enumerate(whole_text) if is_spam(t)])
        #     sorted_idx = sorted_idx[~np.in1d(sorted_idx,spam_idx)]

        self.sorted_idx = sorted_idx[~np.in1d(sorted_idx,missing_time)]
        self.timestamp = timestamp.astype(int)

        print(f'loaded dataset. took {time.perf_counter()-start} ms. Got {len(self.sorted_idx)} items')


        del tokens,embeddings,sentiment_label,sentiment_score

    def __getitem__(self,idx):
        idx = self.sorted_idx[idx]
        ret = {
            'original_index': idx
        }
        if self.use_sentiment:
            ret['sentiment_label'] = self.sentiment_label[idx]
        if self.use_embeddings:
            ret['embedding'] = self.embedding[idx]
        return ret
        
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
