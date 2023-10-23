import networkx as nx
import pandas as pd

from crypto_chatter.config import (
    ES_TWITTER_QUERY,
    ES_HOSTNAME,
    ES_TWITTER_COLUMNS,
    ES_TWITTER_MAPPINGS,
    ES_TWITTER_KEYWORDS,
    DATA_DIR,
    CryptoChatterDataConfig,
)
from .crypto_reply_graph import CryptoReplyGraph


class CryptoTwitterReplyGraph(CryptoReplyGraph):
    graph: nx.DiGraph
    nodes: list[int]
    edges: list[list[int]]
    data: pd.DataFrame
    data_config: CryptoChatterDataConfig
    data_source: str = 'twitter'
    
    def __init__(self, index_name: str) -> None:
        es_query = ES_TWITTER_QUERY
        es_query['query']['bool']['must'] = {
            "simple_query_string": {
                "query": ' '.join(ES_TWITTER_KEYWORDS),
                "fields": [
                    "text",
                    "extended_tweet.full_text"
                ],
            }
        }
        self.data_config = CryptoChatterDataConfig(
            es_hostname=ES_HOSTNAME,
            es_index=index_name,
            es_columns=ES_TWITTER_COLUMNS,
            es_mappings=ES_TWITTER_MAPPINGS,
            es_query=es_query,
            data_source=self.data_source,
            raw_snapshot_dir = DATA_DIR / f'twitter/{index_name}/snapshots',
            graph_dir = DATA_DIR / f'twitter/{index_name}/reply-graph',
            graph_gephi_dir = DATA_DIR / f'twitter/{index_name}/reply-graph/gephi',
            graph_components_dir = DATA_DIR / f'twitter/{index_name}/reply-graph/components',
            graph_stats_file = DATA_DIR / f'twitter/{index_name}/reply-graph/stats.json',
            graph_edges_file = DATA_DIR / f'twitter/{index_name}/reply-graph/edges.json',
            graph_nodes_file = DATA_DIR / f'twitter/{index_name}/reply-graph/nodes.json',
            graph_data_file = DATA_DIR / f'twitter/{index_name}/reply-graph/graph_data.pkl',
            graph_attributes = ['full_text', 'user.id', 'user.followers_count', 'favorite_count', 'quote_count'],
        )
        self.data_config.raw_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.data_config.graph_dir.mkdir(parents=True, exist_ok=True)
        self.data_config.graph_components_dir.mkdir(parents=True, exist_ok=True)
    
    def populate_attributes(self):
        for attr in self.data_config.graph_attributes:
            nx.set_node_attributes(
                self.graph,
                self.data[attr],
                name = attr,
            )
