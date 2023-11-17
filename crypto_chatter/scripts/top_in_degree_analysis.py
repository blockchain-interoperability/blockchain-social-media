import json
import click

from crypto_chatter.graph import CryptoChatterGraph
from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig

# @click.command()
# @click.option('-n', '--top_n', default=1000)
# def run(
#     top_n: int = 100,
# ):

top_n = 1000
dataset = 'twitter:blockchain-interoperability-attacks'
graph_type = 'tweet'
data_config = CryptoChatterDataConfig(dataset)
graph_config = CryptoChatterGraphConfig(data_config, graph_type)
graph = CryptoChatterGraph(
    graph_config,
    data_config,
    columns = ['hashtags']
)

subgraphs = graph.get_top_n_subgraphs('in_degree',top_n)
for i, subgraph in enumerate(subgraphs): 
    keywords = subgraph.get_keywords(10)
    hashtag_count = subgraph.count_hashtags(10)
    subgraph.data.sentiments()
    subgraph.data.embeddings()
    print('#' * 40)
    print(f'Thread: {i}')
    print('keywords:')
    print(json.dumps(keywords, indent=2))
    print('='*40)
    print('hashtags:')
    print(json.dumps(hashtag_count, indent=2))

# if __name__ == "__main__":
#     run()
