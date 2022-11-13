import argparse
import json


from collect_data import cache_index
from tokenizer import tokenize_text
from ngrams import draw_ngrams

parser = argparse.ArgumentParser()

function_mappings = {
    'cache_index': cache_index,
    'tokenize_text': tokenize_text,
    'draw_ngrams_mixed': draw_ngrams,
    'draw_ngrams_emoji': draw_ngrams,
    'draw_ngrams_text': draw_ngrams,
}

parser.add_argument("function", default="cache_index", help="Function to call")
parser.add_argument("--configfile", "-c", default="twitter_config.json", required=False, help="Path to the config file to use.")
# parser.add_argument("--logfile", "-l", default="tmlog.txt", required=False, help="Path to the log file to write to.")

args = parser.parse_args()

function_mappings[args.function](**json.load(open(args.configfile))[args.function])
