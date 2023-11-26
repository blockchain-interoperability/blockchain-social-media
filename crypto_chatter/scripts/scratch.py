from typing import Literal

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import CryptoChatterGraphBuilder
from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig
from crypto_chatter.utils import progress_bar
from crypto_chatter.config.path import BASE_DIR

progress = progress_bar()
progress.start()

dataset = "twitter:blockchain-interoperability-attacks"
data_config = CryptoChatterDataConfig(dataset)
data = CryptoChatterData(
    data_config,
    columns=["hashtags"],
    progress=progress,
)

data.load([data.data_config.clean_text_col])

data.fit_tfidf()

subgraph_type = "quote"
graph_config = CryptoChatterGraphConfig(data_config, f"tweet-{subgraph_type}")
builder = CryptoChatterGraphBuilder(
    data=data,
    graph_config=graph_config,
)

graph = builder.get_graph()

print(f"Data has {len(data):,} rows")

top_n = 100
subgraphs = builder.get_subgraphs(
    graph=graph,
    top_n=top_n,
    kind="centrality",
    centrality="in_degree",
    reachable="undirected",
)


sg = subgraphs[40]

stats = sg.stats(
    data=data,
    include_edge=True,
    include_keywords=True,
    progress=progress
)

output_stream.write(f"## {subgraph_type.capitalize()} Subgraphs\n")
for sg_type, subgraphs in subgraphs.items():
    sg_task = progress.add_task(f"Processing {sg_type}", total=len(subgraphs))
    output_stream.write(f"### By {sg_type}\n")
    for i, sg in enumerate(subgraphs[39:]):
        output_stream.write(f"#### {i:04d}\n\n")
        output_stream.write(f"ID: {sg.id}\n\n")
        output_stream.write(
            graph.stats(
                data=data,
                include_edge=True,
                include_keywords=True,
                recompute=recompute,
                progress=progress
            )
        )
        progress.advance(sg_task)

