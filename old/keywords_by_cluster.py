import pandas as pd
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
# load snapshots
base_path = Path('/data/blockchain-interoperability/blockchain-social-media/twitter-data/')

spam_patterns = [
    ['uniswap is being exploited'],
    ['200k'],
    ['i wish', 'uniswap', 'earlier'],
    ['开云体育'],
    ['世界杯'],
    ['上海']
    # 'http'
]

custom_stop_words = list(ENGLISH_STOP_WORDS.union(["https", '000']+stopwords.words('english')))

sentiment_types = {
    -1: 'negative',
    0: 'neutral',
    1: 'positive'
}

def agg_mask_and(mask_list): 
    agg = mask_list[0]
    for m in mask_list[1:]:
        agg = agg & m
    return agg

def agg_mask_or(mask_list): 
    agg = mask_list[0]
    for m in mask_list[1:]:
        agg = agg | m
    return agg


def load_df():
    # load cluster info
    cluster_ids = json.load(open(base_path/'kmeans_clusters/kmeans_init_clusters.json'))
    index_to_cluster = {
        idx:int(c_id)
        for c_id, idxs in cluster_ids.items() 
        for idx in idxs
    }

    # load sentiment
    text = pd.read_pickle(base_path / 'snapshots/whole_text.pkl').str.lower()
    sentiment = pd.read_pickle(base_path / 'sentiment/transformer/sentiment.pkl')
    sentiment_score = pd.read_pickle(base_path / 'sentiment/transformer/sentiment_score.pkl')

    # turn into combined dataframe
    df = pd.concat([text, sentiment, sentiment_score],axis=1)
    df['cluster_id'] = df.index.map(index_to_cluster)

    # filter out text that contains spam tweets
    # df = df[~df['whole_text'].str.contains('|'.join(spam_patterns))]
    spam_mask = agg_mask_or([agg_mask_and([df.whole_text.str.contains(p) for p in pattern]) for pattern in spam_patterns])
    df = df[~spam_mask]
    print('loaded data!')
    return df



def get_top_keywords(text, top_n, max_df):
    vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features = top_n, max_df = max_df)
    term_scores = vectorizer.fit_transform(text).toarray().sum(0)
    terms = vectorizer.get_feature_names_out()
    keywords = terms[term_scores.argsort()[::-1]].tolist()
    return keywords

# save the keywords

df = load_df()
save_folder = (base_path/ 'sentiment_keywords')
save_folder.mkdir(parents=True, exist_ok=True)

# now do it per cluster
for cluster_id, cluster_tweets in [('whole', df)]+list(df.groupby('cluster_id')):
    cluster_save_folder = save_folder / f'cluster_{cluster_id}'
    cluster_save_folder.mkdir(parents=True, exist_ok=True)

    print(f'Cluster {cluster_id}')

    keywords = get_top_keywords(cluster_tweets['whole_text'], top_n = 100, max_df = 0.9)
    json.dump(keywords, open(cluster_save_folder/f'cluster_keywords.json', 'w'))

    for sentiment, tweets_by_sentiment in cluster_tweets.groupby('sentiment'):
        print(sentiment_types[sentiment])
        keywords = get_top_keywords(tweets_by_sentiment['whole_text'], top_n = 100, max_df = 0.9)
        json.dump(keywords, open(cluster_save_folder/f'{sentiment_types[sentiment]}_keywords.json', 'w'))