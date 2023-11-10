import json
import click

from crypto_chatter.graph import load_graph
from crypto_chatter.graph import CryptoSubGraph
from crypto_chatter.config.default import BlockchainAttackTwitterTweetGraphConfig

@click.command()
@click.option('--top_n', default=100)
def run(
    top_n: int = 100,
):
    data_config = BlockchainAttackTwitterTweetGraphConfig
    graph = load_graph(data_config)
    top_nodes = graph.get_top_n_in_degree(top_n)
    for n in top_nodes: 
        subgraph = CryptoSubGraph(graph, n)
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
