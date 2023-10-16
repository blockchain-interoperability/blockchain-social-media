import networkx as nx
import pandas as pd

from crypto_chatter.config import (
    ES_HOSTNAME,
    ES_REDDIT_QUERY,
    ES_REDDIT_COLUMNS,
    ES_REDDIT_MAPPINGS,
    DATA_DIR,
    # REDDIT_USERNAME,
    # REDDIT_PASSWORD,
    # REDDIT_CLIENT_ID,
    # REDDIT_CLIENT_SECRET,
    CryptoChatterDataConfig,
)
from .crypto_graph import CryptoGraph


class CryptoRedditGraph(CryptoGraph):
    graph: nx.DiGraph
    nodes: list[int]
    edges: list[list[int]]
    data: pd.DataFrame
    data_config: CryptoChatterDataConfig
    data_source: str = 'reddit'
    
    def __init__(self, index_name: str) -> None:
        es_query = ES_REDDIT_QUERY
        # es_query['query']['bool']['must'] = {
        #     "simple_query_string": {
        #         "query": ' '.join(ES_REDDIT_KEYWORDS),
        #         "fields": [
        #             "text",
        #             "extended_tweet.full_text"
        #         ],
        #     }
        # }
        self.data_config = CryptoChatterDataConfig(
            data_source=self.data_source,
            data_dir = DATA_DIR / 'reddit',
            raw_snapshot_dir = DATA_DIR / 'reddit/snapshots',
            graph_dir = DATA_DIR / 'reddit/graph',
            graph_stats_file = DATA_DIR / 'reddit/graph/stats.json',
            graph_edges_file = DATA_DIR / 'reddit/graph/edges.json',
            graph_nodes_file = DATA_DIR / 'reddit/graph/nodes.json',
            graph_data_file = DATA_DIR / 'reddit/graph/graph_data.pkl',
            graph_gephi_file = DATA_DIR / 'reddit/graph/graph.gexf',
            # reddit_username=REDDIT_USERNAME,
            # reddit_password=REDDIT_PASSWORD,
            # reddit_client_id=REDDIT_CLIENT_ID,
            # reddit_client_secret=REDDIT_CLIENT_SECRET,
        )
        self.data_config.raw_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.data_config.graph_dir.mkdir(parents=True, exist_ok=True)
