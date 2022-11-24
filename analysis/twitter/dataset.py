import time
import pandas as pd


from collect_data import load_cache
from tokenizer import load_tokens
from sentiment import get_sentiment

def load_dataset(
    snapshot_path = '',
    token_path = '',
    sentiment_type = '',
):
    start = time.perf_counter
    data = load_cache(snapshot_path)
    tokens = load_tokens(token_path)

    clean_text = [' '.join(t) for t in tokens]
    sentiment = get_sentiment(sentiment_type,clean_text)
    data['clean_text'] = clean_text
    data['tokens'] = tokens
    data['sentiment'] = sentiment
    data['timestamp_ms'] = pd.to_datetime(data['timestamp_ms'],unit='ms')

    print(f'loaded dataset. took {time.perf_counter()-start} ms')


    return data.sort_values('timestamp_ms')


