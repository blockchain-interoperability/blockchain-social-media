import pandas as pd
import json
import time

from crypto_chatter.utils import progress_bar, NodeList, EdgeList
from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.data.load_raw_data import load_raw_data

def load_reply_graph_data(
    data_config: CryptoChatterDataConfig
) -> tuple[pd.DataFrame, NodeList, EdgeList]:
    data_config.graph_dir.mkdir(parents=True, exist_ok=True)
    graph_nodes_file = data_config.graph_dir / 'nodes.json'
    graph_edges_file = data_config.graph_dir / 'edges.json'
    graph_data_file = data_config.graph_dir / 'graph_data.pkl'
    if (
        not graph_nodes_file.is_file() 
        and not graph_edges_file.is_file()
    ):
        df = load_raw_data(data_config)
        has_reply = df[~df['quoted_status.id'].isna()]
        edges_to = []
        edges_from = []

        start = time.time()
        with progress_bar() as progress:
            graph_task = progress.add_task('Constructing edges...', total = len(df))
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

        graph_df = df[df['id'].isin(nodes)]
        graph_df.to_pickle(graph_data_file)

        print(f'Saved node and edge information to {data_config.graph_dir}')

    else:
        start = time.time()
        nodes = json.load(open(graph_nodes_file))
        edges = json.load(open(graph_edges_file))
        print(f'loaded graph edges in {int(time.time() - start)} seconds')

        start = time.time()
        graph_df = pd.read_pickle(graph_data_file)
        print(f'loaded cached graph data in {int(time.time() - start)} seconds')

    return graph_df, nodes, edges
