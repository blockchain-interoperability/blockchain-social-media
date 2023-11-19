import json
import click

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import CryptoChatterGraph
from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig
from crypto_chatter.utils import progress_bar

# @click.command()
# @click.option('-n', '--top_n', default=1000)
# def run(
#     top_n: int = 100,
# ):
progress = progress_bar()
progress.start()

top_n = 100
dataset = 'twitter:blockchain-interoperability-attacks'
graph_type = 'tweet'
data_config = CryptoChatterDataConfig(dataset)
graph_config = CryptoChatterGraphConfig(data_config, graph_type)
data = CryptoChatterData(
    data_config,
    columns = ['hashtags'],
    progress = progress,
)
graph = CryptoChatterGraph(
    data=data,
    graph_config=graph_config,
    progress = progress
)

subgraphs = graph.get_subgraphs(
    subgraph_kind='centrality',
    top_n=top_n,
    centrality='in_degree', 
    reachable='undirected',
)

subgraph_task = progress.add_task('Generating for subgraphs..', total=top_n)
for i, subgraph in enumerate(subgraphs): 
    data.get('embedding', subgraph.nodes)
    data.get('sentiment', subgraph.nodes)
    progress.advance(subgraph_task)
    # subgraph.export_gephi(
    #     node_attributes = ['text','sentiment_positive'],
    #     edge_attributes = [],
    # )

    # keywords = subgraph.get_keywords(10)
    # hashtag_count = subgraph.count_hashtags(10)
    # print('#' * 40)
    # print(f'Thread: {i}')
    # print('keywords:')
    # print(json.dumps(keywords, indent=2))
    # print('='*40)
    # print('hashtags:')
    # print(json.dumps(hashtag_count, indent=2))

progress.stop()

# if __name__ == "__main__":
#     run()
