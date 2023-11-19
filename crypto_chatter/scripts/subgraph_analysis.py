from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import (
    CryptoChatterGraph,
    CryptoChatterSubGraphBuilder
)
from crypto_chatter.config import (
    CryptoChatterDataConfig,
    CryptoChatterGraphConfig
)
from crypto_chatter.utils import progress_bar

top_n = 1000

progress = progress_bar()
progress.start()

dataset = 'twitter:blockchain-interoperability-attacks'
graph_type = 'tweet'
data_config = CryptoChatterDataConfig(dataset)
graph_config = CryptoChatterGraphConfig(data_config, graph_type)
data = CryptoChatterData(
    data_config,
    columns = ['hashtags'],
    progress=progress,
)

graph = CryptoChatterGraph(
    data=data,
    graph_config=graph_config,
    progress=progress,
)

builder = CryptoChatterSubGraphBuilder(
    graph=graph,
    progress=progress,
)

idc_subgraphs = builder.get_subgraphs(
    subgraph_kind='centrality',
    top_n=top_n,
    centrality='in_degree', 
    reachable='undirected',
)
wcc_subgraphs = builder.get_subgraphs(
    subgraph_kind='component',
    component='weak',
    top_n=top_n,
)

subgraphs = idc_subgraphs+wcc_subgraphs
subgraph_task = progress.add_task('Generating for subgraphs..', total=len(subgraphs))
for subgraph in idc_subgraphs+wcc_subgraphs:
    data.get('embedding', subgraph.nodes)
    data.get('sentiment', subgraph.nodes)
    progress.advance(subgraph_task)

progress.stop()
