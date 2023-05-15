from pathlib import Path
import argparse
import json

from utils.collect_data import cache_index
from utils.tokenizer import tokenize_text
from utils.ngrams import draw_ngrams
from utils.embeddings import get_sbert_embeddings
from utils.sentiment import get_sentiment
from utils.autoencoders import train_encoder


parser = argparse.ArgumentParser()

op_mappings = {
    'cache_index': cache_index,
    'tokenize_text': tokenize_text,
    'get_sbert_embeddings': get_sbert_embeddings,
    'get_sentiment': get_sentiment,
    'train_encoder_linear': train_encoder,
    # 'get_umap_embeddings': get_umap_embeddings
    # 'run_kmeans': run_kmeans,
    # 'parse_clusters': parse_clusters
}

parser.add_argument("operation", default="cache_index", help="Function to call")
parser.add_argument("--configfile", "-c", default="config.json", required=False, help="Path to the config file to use.")


args = parser.parse_args()

configuration = json.load(open(args.configfile))

# print(dict(args))
op_mappings[args.operation](**configuration)


