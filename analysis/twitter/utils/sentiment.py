from nltk.sentiment.vader import SentimentIntensityAnalyzer
import transformers
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

VADER_SENTIMENT_LABEL_MAPPINGS = {
    'POS': 1,
    'NEG': -1,
    'NEU': 0
}

def vader_sent_score(text,analyzer):
    scores = analyzer.polarity_scores(text)
    tag,score = max(scores.items(),key=lambda v: v[1] if v[0]!='compound' else -10 )
    # return if scores['psos']
    # return scores['pos'] > scores['neg'] 
    return VADER_SENTIMENT_LABEL_MAPPINGS[tag.upper()],score


# 0	negative
# 1	neutral
# 2	positive
TRNASFORMER_SENTIMENT_LABEL_MAPPINGS = {
    'LABEL_0': -1,
    'LABEL_1': 0,
    'LABEL_2': 1,
}

def transformer_sent_score(text,analyzer):
    # {'label': 'POS', 'score': 0.5677285194396973}
    scores = analyzer(text)[0]
    return TRNASFORMER_SENTIMENT_LABEL_MAPPINGS[scores['label'].upper()],scores['score']
    
    


ANALYZERS = {
    'vader': SentimentIntensityAnalyzer,
    'transformer':lambda: transformers.pipeline(
        'sentiment-analysis',
        model = 'cardiffnlp/twitter-roberta-base-sentiment',
        tokenizer='cardiffnlp/twitter-roberta-base-sentiment',
        truncation=True,
        max_length=512, 
        device=0,
    )
}

SENTIMENT_MAPPING = {
    'vader': vader_sent_score,
    'transformer': transformer_sent_score
}


def get_sentiment(
    sentiment_type = 'vader',
    snapshot_path = '',
    sentiment_path = '',
):
    sentiment_path = Path(sentiment_path) / f'{sentiment_type}'
    snapshot_path = Path(snapshot_path)
    sentiment_path.mkdir(parents=True,exist_ok=True)

    snap_files = sorted(snapshot_path.glob('*.pkl'))
    sent_files = sorted(sentiment_path.glob('*.pkl'))
    sentiment = []
    for partial in tqdm(snap_files, desc= f'{sentiment_type} sentiment scores..',leave =False):
        analyzer = ANALYZERS[sentiment_type]()
        df = pd.read_pickle(partial)
        batch_sentiment = [
            SENTIMENT_MAPPING[sentiment_type](t,analyzer) 
            for t in tqdm(df.whole_text,desc='batch..',leave=False)
        ]
        pickle.dump(batch_sentiment,open(sentiment_path/partial.name,'wb'))
        del df,batch_sentiment,analyzer



def load_sentiment(sentiment_path):
    sentiment_labels = []
    sentiment_scores = []
    files = sorted(Path(sentiment_path).glob('*.pkl'))
    for f in tqdm(files,desc='loading sentiment..',leave=False):
        batch_labels,batch_scores= zip(*pickle.load(open(f,'rb')))
        sentiment_labels += batch_labels
        sentiment_scores += batch_scores
        # all_sentiments += pickle.load(open(f,'rb'))
    
    return np.array(sentiment_labels),np.array(sentiment_scores)
        