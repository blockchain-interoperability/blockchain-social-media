import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle
from dataclasses import dataclass

from crypto_chatter.utils.types import TextList

from .text import clean_text, is_spam, STOP_WORDS

@dataclass
class TfidfConfig:
    random_seed:int = 0
    random_size:int = int(1e6)
    ngram_range:tuple[int,int] = (1, 1)
    max_df:float|int = 1.0
    min_df:int = 1
    max_features:int = 10000

    def __repr__(self):
        return f"{self.random_seed}_{self.random_size}_{self.ngram_range}_{self.max_df}_{self.min_df}_{self.max_features}"

def fit_tfidf(
    text: TextList,
    cache_dir: Path,
    config: TfidfConfig,
) -> TfidfVectorizer:
    save_file = cache_dir / f"tfidf/{config}.pkl"
    save_file.parent.mkdir(parents=True, exist_ok=True)

    if not save_file.is_file():
        print('fitting tfidf..')
        start = time.time()
        rng = np.random.RandomState(config.random_seed)
        # first filter out spam
        not_spam = [t for t in text if not is_spam(t)]
        # Then get random indices
        if config.random_size > 0:
            random_idxs = rng.permutation(np.arange(len(not_spam)))[:config.random_size]
        else:
            random_idxs = np.arange(len(not_spam))
        subset = [clean_text(not_spam[i]) for i in random_idxs]
        tfidf = TfidfVectorizer(
            stop_words=STOP_WORDS,
            ngram_range=config.ngram_range,
            max_df=config.max_df,
            min_df=config.min_df,
            max_features=config.max_features,
        )
        tfidf.fit(subset)
        pickle.dump(tfidf, open(save_file,"wb"))
        print(f"computed tfidf and saved in {time.time() - start:.2f} seconds")
    else:
        print('loading tfidf..')
        tfidf = pickle.load(open(save_file, "rb"))
    return tfidf

def get_tfidf(
    texts: TextList,
    tfidf: TfidfVectorizer,
) -> dict[str,float]:
    terms = tfidf.get_feature_names_out()
    vecs = tfidf.transform(texts)
    tfidf_scores = vecs.toarray().sum(0)
    sorted_idxs = tfidf_scores.argsort()[::-1]
    keywords = list(terms[sorted_idxs])
    keyword_scores = tfidf_scores[sorted_idxs]
    return list(zip(keywords, keyword_scores))
