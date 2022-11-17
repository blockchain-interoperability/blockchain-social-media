import argparse
import json

from collect_data import cache_index
from tokenizer import tokenize_text
from ngrams import draw_ngrams
from embeddings import get_bert_embeddings,get_umap_embeddings

parser = argparse.ArgumentParser()

function_mappings = {
    'cache_index': cache_index,
    'tokenize_text': tokenize_text,
    'draw_ngrams_mixed': draw_ngrams,
    'draw_ngrams_emoji': draw_ngrams,
    'draw_ngrams_text': draw_ngrams,
    'get_bert_embeddings': get_bert_embeddings,
    'get_umap_embeddings': get_umap_embeddings
}

parser.add_argument("function", default="cache_index", help="Function to call")
parser.add_argument("--configfile", "-c", default="twitter_config.json", required=False, help="Path to the config file to use.")

args = parser.parse_args()

function_mappings[args.function](**json.load(open(args.configfile))[args.function])
