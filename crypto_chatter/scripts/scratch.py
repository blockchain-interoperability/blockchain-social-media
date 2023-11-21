from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import CryptoChatterGraphBuilder
from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig
from crypto_chatter.utils import progress_bar

import numpy as np
from collections import Counter

with progress_bar() as progress:
    dataset = "twitter:blockchain-interoperability-attacks"
    graph_type = "tweet"
    data_config = CryptoChatterDataConfig(dataset)
    graph_config = CryptoChatterGraphConfig(data_config, graph_type)
    data = CryptoChatterData(
        data_config=data_config,
        columns=["hashtags"],
        progress=progress,
    )
    builder = CryptoChatterGraphBuilder(
        data=data,
        graph_config=graph_config,
        progress=progress,
    )

    graph = builder.get_graph()
    subgraphs = builder.get_subgraphs(
        graph=graph,
        kind="centrality",
        top_n=10,
        centrality="in_degree",
        reachable="undirected",
    )

quit()

sg = subgraphs[0]
shortest_path = sg.shortest_path(
    source=sg.source,
    kind="reversed",
)

neighbor_by_depth = dict()
for n, path in shortest_path.items():
    depth = path["length"]
    neighbor_by_depth.setdefault("depth", []).append(n)

for d, nodes in neighbor_by_depth.items():
    sent = data.get("sentiment", nodes)
    sent_count = Counter([s.overall() for s in sent])
    sent_sum = np.stack([s.to_list() for s in sent]).sum(0)
    by_sum = ["positive", "negative", "neutral"][sent_sum.argmax()]

    print(f"Depth {d}: {len(nodes)}")
    print(
        f"Positive: {sent_count['positive']}/{len(nodes)} -- {sent_count['positive']/len(nodes):.2f}%"
    )
    print(
        f"Negative: {sent_count['negative']}/{len(nodes)} -- {sent_count['negative']/len(nodes):.2f}%"
    )
    print(
        f"Neutral: {sent_count['neutral']}/{len(nodes)} -- {sent_count['neutral']/len(nodes):.2f}%"
    )
    print(f"By Sum: {by_sum}")
