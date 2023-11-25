import pandas as pd

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import (
    CryptoChatterGraphBuilder
)
from crypto_chatter.config import (
    CryptoChatterDataConfig,
    CryptoChatterGraphConfig
)
from crypto_chatter.utils import progress_bar

top_n = 100

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

print(f"data has {len(data):,} rows")
data.fit_tfidf()

builder = CryptoChatterGraphBuilder(
    data=data,
    graph_config=graph_config,
    progress=progress,
)

graph = builder.get_graph()
print(f"Tweet Graph has {len(graph.nodes):,} nodes and {len(graph.edges):,} edges")

subgraphs = builder.get_subgraphs(
    graph=graph,
    top_n=100,
    kind="centrality",
    centrality="in_degree",
    reachable="undirected",
) + builder.get_subgraphs(
    graph=graph,
    top_n=100,
    kind="component",
    component="weak",
) + builder.get_subgraphs(
    graph=graph,
    top_n=100,
    kind="community",
    community="louvain",
    random_seed=0
)

for i, sg in enumerate(subgraphs):
    kws = pd.DataFrame(
        sg.get_keywords(
            data=data,
        ),
        columns=["keyword", "score"],
    )

    edge_sims = pd.DataFrame(
        [
            dict(
                node_from=n1,
                node_to=n2,
                sim=sim,
            )
            for (n1, n2), sim in sg.get_edge_attribute(
                data=data, kind="emb_cosine_sim"
            ).items()
        ]
    )

    print("#" * 80)
    print(f"SG {i} -- {sg.id}")
    print("=" * 80)
    print("EDGE SIMILARITIES")
    print("  AVG:",edge_sims["sim"].mean())
    print("  STD:",edge_sims["sim"].std())
    print("  MAX:",edge_sims["sim"].max())
    print("  MIN:",edge_sims["sim"].min())
    print("=" * 80)
    print("KEYWORDS")
    print(kws.iloc[:10].to_markdown(index=False))

progress.stop()
