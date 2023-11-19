from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import (
    CryptoChatterGraph,
    CryptoChatterSubGraphBuilder
)
from crypto_chatter.config import (
    CryptoChatterDataConfig,
    CryptoChatterGraphConfig
)

dataset = 'twitter:blockchain-interoperability-attacks'
graph_type = 'tweet'
data_config = CryptoChatterDataConfig(dataset)
graph_config = CryptoChatterGraphConfig(data_config, graph_type)
data = CryptoChatterData(
    data_config,
    columns = ['hashtags'],
)

graph = CryptoChatterGraph(
    data=data,
    graph_config=graph_config,
)

builder = CryptoChatterSubGraphBuilder(graph)

subgraphs = builder.get_subgraphs(
    subgraph_kind='centrality',
    top_n=10,
    centrality='in_degree', 
    reachable='undirected',
)


