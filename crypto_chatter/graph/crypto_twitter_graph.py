import networkx as nx
import pandas as pd

from crypto_chatter.config import (
    ES_TWITTER_QUERY,
    ES_HOSTNAME,
    ES_TWITTER_INDEXNAME,
    ES_TWITTER_COLUMNS,
    ES_TWITTER_KEYWORDS,
    DATA_DIR,
    CryptoChatterDataConfig,
)
from .crypto_graph import CryptoGraph


class CryptoTwitterGraph(CryptoGraph):
    graph: nx.DiGraph
    nodes: list[int]
    edges: list[list[int]]
    data: pd.DataFrame
    data_config: CryptoChatterDataConfig
    data_source: str = 'twitter'
    
    def __init__(self) -> None:
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
            es_index=ES_TWITTER_INDEXNAME,
            es_columns=ES_TWITTER_COLUMNS,
            es_query=es_query,
            data_source=self.data_source,
            data_dir = DATA_DIR / 'twitter',
            raw_snapshot_dir = DATA_DIR / 'twitter/snapshots',
            graph_dir = DATA_DIR / 'twitter/graph',
            graph_stats_file = DATA_DIR / 'twitter/graph/stats.json',
            graph_edges_file = DATA_DIR / 'twitter/graph/edges.json',
            graph_nodes_file = DATA_DIR / 'twitter/graph/nodes.json',
            graph_data_file = DATA_DIR / 'twitter/graph/graph_data.pkl',
        )
        self.data_config.raw_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.data_config.graph_dir.mkdir(parents=True, exist_ok=True)
