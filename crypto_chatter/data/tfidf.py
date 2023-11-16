import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import pickle

from .utils import preprocess_text, is_spam
from crypto_chatter.config import CryptoChatterDataConfig



def fit_tfidf(
    # self,
    text: list[str]|np.ndarray|pd.Series,
    data_config: CryptoChatterDataConfig,
    random_seed:int = 0,
    random_size:int = 1000000,
    ngram_range:tuple[int,int] = (1, 1),
    max_df:float|int = 1.0,
    min_df:float|int = 1,
    max_features:int = 10000,
) -> TfidfVectorizer:
    tfidf_settings = f"{random_seed}_{random_size}_{ngram_range}_{max_df}_{min_df}_{max_features}"
    save_file = data_config.data_dir / f"tfidf/{tfidf_settings}.pkl"
    save_file.parent.mkdir(parents=True, exist_ok=True)

    stop_words = list(ENGLISH_STOP_WORDS | set(['https', '@']))

    if not save_file.is_file():
        start = time.time()
        rng = np.random.RandomState(random_seed)
        # first filter out spam
        not_spam = [t for t in text if not is_spam(t)]
        # Then get random indices
        random_idxs = rng.permutation(np.arange(len(not_spam)))[:random_size]
        subset = [preprocess_text(not_spam[i]) for i in random_idxs]
        tfidf = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
        )
        tfidf.fit(subset)
        pickle.dump(tfidf, open(save_file,"wb"))
        print(f"computed tfidf and saved in {int(time.time() - start)} seconds")
    else:
        tfidf = pickle.load(open(save_file, "rb"))
    return tfidf

def get_tfidf(
    text: list[str]|np.ndarray|pd.Series,
    tfidf: TfidfVectorizer,
) -> tuple[list[str],list[str]]: 
    terms = tfidf.get_feature_names_out()
    vecs = tfidf.transform(text)
    tfidf_scores = vecs.toarray().sum(0)
    sorted_idxs = tfidf_scores.argsort()[::-1]
    keywords = terms[sorted_idxs]
    keyword_scores = tfidf_scores[sorted_idxs]
    return keywords, keyword_scores
