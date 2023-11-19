from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import CryptoChatterGraph
from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig

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

subgraphs = graph.get_subgraphs(
    subgraph_kind='centrality',
    top_n=100,
    centrality='in_degree', 
    reachable='undirected',
)

for i, s in enumerate(subgraphs):
    print(i, len(s.nodes))
