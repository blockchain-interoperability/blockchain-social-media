import pandas as pd
import json
import time

from crypto_chatter.utils import progress_bar, NodeList, EdgeList
from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig
from crypto_chatter.data import CryptoChatterData, load_snapshots

def load_reply_graph_data(
    data_config: CryptoChatterDataConfig,
    graph_config: CryptoChatterGraphConfig,
) -> tuple[CryptoChatterData, NodeList, EdgeList]:
    graph_config.graph_dir.mkdir(parents=True, exist_ok=True)
    graph_nodes_file = graph_config.graph_dir / 'nodes.json'
    graph_edges_file = graph_config.graph_dir / 'edges.json'

    data = CryptoChatterData(data_config, ['id','quoted_status.id'])
    if (
        not graph_nodes_file.is_file() 
        and not graph_edges_file.is_file()
    ):
        has_reply = data[~data['quoted_status.id'].isna()]
        edges_to = []
        edges_from = []

        start = time.time()
        with progress_bar() as progress:
            graph_task = progress.add_task('Constructing edges...', total = len(data))
            for tweet_id, reply_id in zip(
                has_reply['quoted_status.id'].values,
                has_reply['id'].values
            ):
                if not pd.isna(reply_id):
                    edges_from += [int(reply_id)]
                    edges_to += [int(tweet_id)]
                progress.update(graph_task, advance =1)

        nodes = list(set(edges_to) | set(edges_from))
        edges = list(zip(edges_from, edges_to))
        print('saved graph data to cache')

        json.dump(
            nodes,
            open(graph_nodes_file, 'w')
        )
        json.dump(
            edges,
            open(graph_edges_file, 'w')
        )
        
        print(f'Constructed graph with {len(nodes):,} nodes and {len(edges_to):,} edges in {int(time.time() - start)} seconds')
        print(f'Saved node and edge information to {graph_config.graph_dir}')

    else:
        start = time.time()
        nodes = json.load(open(graph_nodes_file))
        edges = json.load(open(graph_edges_file))
        print(f'loaded graph edges in {int(time.time() - start)} seconds')

    return data, nodes, edges
