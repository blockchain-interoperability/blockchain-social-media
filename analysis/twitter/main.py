import argparse
import json

from collect_data import cache_index
from tokenizer import tokenize_text
from ngrams import draw_ngrams
from embeddings import create_sbert_embeddings
from sentiment import get_sentiment
from autoencoders import train_encoder
from clustering import run_kmeans

parser = argparse.ArgumentParser()

function_mappings = {
    'cache_index': cache_index,
    'tokenize_text': tokenize_text,
    'draw_ngrams_mixed_spam': draw_ngrams,
    'draw_ngrams_emoji_spam': draw_ngrams,
    'draw_ngrams_text_spam': draw_ngrams,
    'draw_ngrams_mixed_nospam': draw_ngrams,
    'draw_ngrams_emoji_nospam': draw_ngrams,
    'draw_ngrams_text_nospam': draw_ngrams,
    'create_sbert_embeddings': create_sbert_embeddings,
    'get_vader_sentiment': get_sentiment,
    'get_trans_sentiment': get_sentiment,
    # 'get_umap_embeddings': get_umap_embeddings
    'train_encoder_linear': train_encoder,
    'run_kmeans': run_kmeans,
    # 'parse_clusters': parse_clusters
}

parser.add_argument("function", default="cache_index", help="Function to call")
parser.add_argument("--configfile", "-c", default="twitter_config.json", required=False, help="Path to the config file to use.")

args = parser.parse_args()

function_mappings[args.function](**json.load(open(args.configfile))[args.function])
