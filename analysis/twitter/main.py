from pathlib import Path
import argparse
import json

from utils.collect_data import cache_index
from utils.tokenizer import tokenize_text
from utils.ngrams import draw_ngrams
from utils.embeddings import get_transformer_embeddings
from utils.sentiment import get_sentiment
from utils.autoencoders import train_encoder
from utils.clustering import run_kmeans

DEFAULT_PATH = Path('/data/blockchain-interoperability/blockchain-social-media/twitter-data/')

parser = argparse.ArgumentParser()

op_mappings = {
    'cache_index': cache_index,
    'tokenize_text': tokenize_text,
    'draw_ngrams_mixed_spam': draw_ngrams,
    'draw_ngrams_emoji_spam': draw_ngrams,
    'draw_ngrams_text_spam': draw_ngrams,
    'draw_ngrams_mixed_nospam': draw_ngrams,
    'draw_ngrams_emoji_nospam': draw_ngrams,
    'draw_ngrams_text_nospam': draw_ngrams,
    'get_transformer_embeddings': get_transformer_embeddings,
    'get_vader_sentiment': get_sentiment,
    'get_trans_sentiment': get_sentiment,
    # 'get_umap_embeddings': get_umap_embeddings
    'train_encoder_linear': train_encoder,
    'run_kmeans': run_kmeans,
    # 'parse_clusters': parse_clusters
}

parser.add_argument("operation", default="cache_index", help="Function to call")
parser.add_argument("--configfile", "-c", default="config.json", required=False, help="Path to the config file to use.")



args = parser.parse_args()

configuration = json.load(open(args.configfile))

# print(dict(args))
op_mappings[args.operation](**configuration)


