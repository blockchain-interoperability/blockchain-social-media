from typing import Literal
import click

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import CryptoChatterGraphBuilder
from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig
from crypto_chatter.utils import progress_bar
from crypto_chatter.config.path import REPORT_DIR

tweet_graph_node_attributes=[
    "text",
    # "sentiment_positive",
    # "sentiment_negative",
    # "sentiment_neutral",
    "sentiment_composite",
    "retweet_count",
    "favorite_count",
    "quote_count",
    "reply_count",
]
tweet_graph_edge_attributes=[
    # "emb_cosine_sim"
]

user_graph_node_attributes=[
    "followers_count",
    "friends_count",
    "avg_retweet_count",
    "avg_quote_count",
    "avg_reply_count",
    "avg_favorite_count",
    "total_retweet_count",
    "total_quote_count",
    "total_reply_count",
    "total_favorite_count",
    # "avg_sentiment_positive",
    # "avg_sentiment_negative",
    # "avg_sentiment_neutral",
]
user_graph_edge_attributes=[
    "total_quote_count",
    "total_reply_count",
    "avg_quote_count",
    "avg_reply_count",
]

@click.command()
@click.argument('graph_kind')
@click.option('--top_n', default=10)
def subgraph_stats(
    graph_kind: Literal['tweet-reply','tweet-quote', 'user-reply','user-quote'],
    top_n: int,
) -> None:

    if 'tweet' in graph_kind:
        node_attributes = tweet_graph_node_attributes
        edge_attributes = tweet_graph_edge_attributes
    elif 'user' in graph_kind:
        node_attributes = user_graph_node_attributes
        edge_attributes = user_graph_edge_attributes
    else:
        raise ValueError(f"graph_kind must be one of ['tweet-reply','tweet-quote', 'user-reply','user-quote']")

    progress = progress_bar()
    progress.start()

    dataset = "twitter:blockchain-interoperability-attacks"
    data_config = CryptoChatterDataConfig(dataset)
    data = CryptoChatterData(
        data_config,
        progress=progress,
    )

    data.load([data.data_config.clean_text_col])
    data.fit_tfidf()

    # graph_kind = f"tweet-{tweet_graph_kind}"
    graph_config = CryptoChatterGraphConfig(
        data_config=data_config,
        graph_kind=graph_kind,
    )
    builder = CryptoChatterGraphBuilder(
        data=data,
        graph_config=graph_config,
    )

    graph = builder.get_graph()

    print(f"Data has {len(data):,} rows")

    output_stream = open(REPORT_DIR / f"{graph_kind}-graph.md", "w")
    output_stream.write(f"# {graph_kind.capitalize()} Graph Stats:\n\n")
    output_stream.write(
        graph.stats(
            data=data, 
            progress=progress,
        )
    )

    # small_graph = builder.random_reduce(
    #     graph=graph,
    #     random_nodes_size=int(8e5),
    #     random_seed=0,
    # )

    # small_graph.export_gephi(
    #     data=data,
    #     node_attributes=node_attributes,
    #     edge_attributes=edge_attributes,
    #     progress=progress,
    # )

    # output_stream.write(
    #     f"## {graph_kind.capitalize()} Graph Random Edge Sample Stats:\n\n"
    # )
    # output_stream.write(f"[gephi file]({str(small_graph.cache_dir/'graph.gexf')})\n")

    # output_stream.write(
    #     small_graph.stats(
    #         data=data,
    #         node_attributes=node_attributes,
    #         edge_attributes=edge_attributes,
    #         # include_keywords=True,
    #         progress=progress
    #     )
    # )

    if 'tweet' in graph_kind:
        subgraphs = {
            # "In Degree Centrality": builder.get_subgraphs(
            #     graph=graph,
            #     top_n=top_n,
            #     kind="centrality",
            #     centrality="in_degree",
            #     reachable="undirected",
            # ),
            "Weakly Connnected Components": builder.get_subgraphs(
                graph=graph,
                top_n=top_n,
                kind="component",
                component="weak",
            ),
            # "Louvain Community": builder.get_subgraphs(
            #     graph=graph,
            #     top_n=top_n,
            #     kind="community",
            #     community="louvain",
            #     random_seed=0,
            # ),
        }

    elif 'user' in graph_kind:
        subgraphs = {
            "Strongly Connected Components": builder.get_subgraphs(
                graph=graph,
                top_n=top_n,
                kind="component",
                component="strong",
            ),
        }
    else:
        raise ValueError(f"graph_kind must be one of ['tweet-reply','tweet-quote', 'user-reply','user-quote']")

    output_stream.write(f"## {graph_kind.capitalize()} Subgraphs\n")
    for sg_type, subgraphs in subgraphs.items():
        sg_task = progress.add_task(f"Processing {sg_type}", total=len(subgraphs))
        output_stream.write(f"### By {sg_type}\n")
        for i, sg in enumerate(subgraphs):
            print(f"Subgraph {sg_type} -- {i} has {len(sg.nodes):,} nodes and {len(sg.edges):,} edges")
            sg.export_gephi(
                data=data,
                node_attributes=node_attributes,
                edge_attributes=edge_attributes,
                progress=progress,
            )

            output_stream.write(f"#### {i:04d}\n\n")
            output_stream.write(f"ID: {sg.id}\n\n")
            output_stream.write(f"[gephi file]({str(sg.cache_dir/'graph.gexf')})\n")
            output_stream.write(
                sg.stats(
                    data=data,
                    node_attributes=node_attributes,
                    edge_attributes=edge_attributes,
                    include_keywords=('tweet' in graph_kind),
                    progress=progress
                )
            )
            progress.advance(sg_task)

    progress.stop()
    output_stream.close()

if __name__ == "__main__":
    subgraph_stats()
