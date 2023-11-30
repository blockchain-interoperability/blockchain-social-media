from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import CryptoChatterGraphBuilder
from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig
from crypto_chatter.utils import progress_bar

progress = progress_bar()
progress.start()

dataset = "twitter:blockchain-interoperability-attacks"
data_config = CryptoChatterDataConfig(dataset)
data = CryptoChatterData(
    data_config,
    progress=progress,
)

data.load([data.data_config.clean_text_col])

graph_kind = f"tweet-quote"
graph_config = CryptoChatterGraphConfig(
    data_config=data_config,
    graph_kind=graph_kind,
)
builder = CryptoChatterGraphBuilder(
    data=data,
    graph_config=graph_config,
)

graph = builder.get_graph()


small_graph = builder.random_reduce(
    graph=graph,
    random_edges_size=int(8e5),
    random_seed=0,
)
len(small_graph.nodes), len(small_graph.edges)

# subgraphs = builder.get_subgraphs(
#     graph=graph,
#     top_n=10,
#     kind="centrality",
#     centrality="in_degree",
#     reachable="undirected",
# )

# sg = subgraphs[9]

has_text = data[data.data_config.text_col].notna()
from_edge_valid = data[graph_config.edge_from_col].notna()
to_edge_valid = data[graph_config.edge_to_col].notna()
valid = data[(has_text & from_edge_valid & to_edge_valid)]


all_edges_to = valid[graph_config.edge_to_col].astype(int)
all_edges_from = valid[graph_config.edge_from_col].astype(int)

nodes_in_ids = all_edges_to.isin(all_edges_from)


edges_from = all_edges_to[nodes_in_ids]
edges_to = all_edges_from[nodes_in_ids]

nodes = list(set(edges_to) | set(edges_from))
edges = list(zip(edges_from, edges_to))


import re
