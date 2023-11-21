import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

nlp = spacy.load("en_core_web_sm")

def extract_hashtags(
    tweet: str
) -> list[str]:
    return [w[1:] for w in tweet.split() if w.startswith("#")]

def preprocess_text(text:str) -> str:
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = '' if t.startswith('0x') else t
        new_text += [t]
    return " ".join(new_text).strip()

def clean_text(text:str) -> str:
    new_text = []
    for t in text.lower().split(" "):
        if not (
            (t.startswith('@') and len(t))
            or t.startswith('http')
            or t.startswith('0x')
        ):
            new_text += [t]
    return " ".join([w.lemma_ for w in nlp(" ".join(new_text).strip())]) 

def is_spam(text:str) -> bool:
    if 'i wish i discovered this earlier' in text.lower():
        return True
    if 'uniswap is being exploited by this dude' in text.lower():
        return True
    if 'more than $200k so far' in text.lower():
        return True

    return False

STOP_WORDS = list(ENGLISH_STOP_WORDS | set(['https', '@']))
