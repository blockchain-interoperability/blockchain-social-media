import json
import click

from crypto_chatter.graph import CryptoChatterSubGraph
from crypto_chatter.config import load_default_data_config, load_default_graph_config

@click.command()
@click.option('--top_n', default=100)
def run(
    top_n: int = 100,
):
    dataset = 'twitter:blockchain-interoperability-attacks'
    graph_type = 'tweet'
    data_config = load_default_data_config(dataset)
    graph_config = load_default_graph_config(dataset, graph_type)
    graph = CryptoChatterTwitterTweetGraph(graph_config, data_config)

    top_nodes = graph.get_top_n_in_degree(top_n)
    for n in top_nodes: 
        subgraph = CryptoChatterSubGraph(graph, n)
        keywords = subgraph.get_keywords(10)
        hashtag_count = subgraph.count_hashtags(10)
        print('#' * 40)
        print('keywords:')
        print(json.dumps(keywords, indent=2))
        print('='*40)
        print('hashtags:')
        print(json.dumps(hashtag_count, indent=2))

if __name__ == "__main__":
    run()
