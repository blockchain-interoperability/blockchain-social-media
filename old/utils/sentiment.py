from nltk.sentiment.vader import SentimentIntensityAnalyzer
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# 0	negative
# 1	neutral
# 2	positive

VADER_SENTIMENT_LABEL_MAPPINGS = {
    'POS': 1,
    'NEG': -1,
    'NEU': 0
}

TRNASFORMER_SENTIMENT_LABEL_MAPPINGS = {
    'LABEL_0': -1,
    'LABEL_1': 0,
    'LABEL_2': 1,
}
    
class SentimentAnalyzer: 
    def __init__(self,sentiment_type):
        self.sentiment_type = sentiment_type

        if sentiment_type == 'vader': 
            self.analyzer = SentimentIntensityAnalyzer()
            self.analyze = self.vader_sent_score

        elif sentiment_type == 'transformer':
            # self.analyzer = {
            #     'tokenizer': AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment'),
            #     'model': AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment').cuda()
            # }

            # sentiment_tokenizer = 
            # sentiment_model = 
            # sentiment_model.to(device)
            
            
            
            
            self.analyzer = transformers.pipeline(
                'sentiment-analysis',
                model = 'cardiffnlp/twitter-roberta-base-sentiment',
                tokenizer='cardiffnlp/twitter-roberta-base-sentiment',
                truncation=True,
                max_length=512, 
                device=0,
            )
            self.analyze = self.transformer_sent_score

    def vader_sent_score(self,textlist):
        return [
            self._vader_sent_score(text)
            for text in textlist
        ]

    def _vader_sent_score(self,text):
        # for individual text
        scores = self.analyzer.polarity_scores(text)
        tag,score = max(scores.items(),key=lambda v: v[1] if v[0]!='compound' else -10 )
        return {
            'sentiment': VADER_SENTIMENT_LABEL_MAPPINGS[tag.upper()], 
            'sentiment_score': score
        }

    def transformer_sent_score(self,textlist):
        # {'label': 'POS', 'score': 0.5677285194396973}
        scores = self.analyzer(textlist)
        return [
            {
            'sentiment': TRNASFORMER_SENTIMENT_LABEL_MAPPINGS[one_score['label'].upper()], 
            'sentiment_score': one_score['score']
            }
            for one_score in scores
        ]


def get_sentiment(
    sentiment_type = 'vader',
    snapshot_path = '',
    sentiment_path = '',
    batch_size = 1000,
    **kwargs
):
    sentiment_path = Path(sentiment_path) / f'{sentiment_type}'
    snapshot_path = Path(snapshot_path)
    sentiment_path.mkdir(parents=True,exist_ok=True)
    whole_text = pd.read_pickle(snapshot_path / 'whole_text.pkl')
    analyzer = SentimentAnalyzer(sentiment_type)

    sentiment = []
    for batch_start in tqdm(range(0,len(whole_text),batch_size),desc='generating sentiment for text...'):
        sentiment += analyzer.analyze(whole_text[batch_start:batch_start +batch_size].tolist())
    sentiment = pd.DataFrame(sentiment)
    
    sentiment['sentiment'].to_pickle(sentiment_path/'sentiment.pkl')
    sentiment['sentiment_score'].to_pickle(sentiment_path/'sentiment_score.pkl')
    return sentiment

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
        