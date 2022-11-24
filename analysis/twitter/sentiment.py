from nltk.sentiment.vader import SentimentIntensityAnalyzer
import transformers
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import pandas as pd
from collect_data import load_cache

def vader_sent_score(text,analyzer):
    scores = analyzer.polarity_scores(text)
    return scores['pos'] > scores['neg'] 


TRANSFORMER_LABEL_MAPPINGS = {
    'POS': 1,
    'NEG': -1,
    'NEU': 0
}

def transformer_sent_score(text,analyzer):
    # [{'label': 'POS', 'score': 0.5677285194396973}]
    scores = analyzer.polarity_scores(text)

    return scores['pos'] > scores['neg'] 
    


ANALYZERS = {
    'vader': SentimentIntensityAnalyzer,
    'transformer':lambda _: transformers.pipeline('sentiment-analysis','finiteautomata/bertweet-base-sentiment-analysis')
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
    sentiment_path = Path(sentiment_path) / f'{sentiment_type}.pkl'
    sentiment_path.parent.mkdir(parents=True,exist_ok=True)
    if sentiment_path.is_file():
        sentiment = pickle.load(open(sentiment_path,'rb'))
    else:
        # snapshots = load_cache(snapshot)
        sentiment = []
        analyzer = ANALYZERS[sentiment_type]()
        for partial in tqdm(sorted(Path(snapshot_path).glob('*.pkl')), desc= f'{sentiment_type} sentiment scores..',leave =False):
            df = pd.read_pickle(partial)
            sentiment += [
                SENTIMENT_MAPPING[sentiment_type](t,analyzer) 
                for t in df.whole_text
            ]
            del df
        pickle.dump(sentiment,open(sentiment_path,'wb'))
    
    return sentiment