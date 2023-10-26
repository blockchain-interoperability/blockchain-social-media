import pandas as pd
import json
import time

from crypto_chatter.utils import progress_bar, NodeList, EdgeList
from crypto_chatter.config import CryptoChatterDataConfig
from .load_raw_data import load_raw_data

def load_user_graph_edges(
    data_config: CryptoChatterDataConfig
) -> tuple[NodeList, EdgeList]:
    if (
        not data_config.graph_nodes_file.is_file() 
        and not data_config.graph_edges_file.is_file()
    ):
        df = load_raw_data(data_config)
        has_quoter = df[~df['user.id'] & ~df['quoted_status.user.id'].isna()]
        edges_to = []
        edges_from = []

        start = time.time()
        with progress_bar() as progress:
            graph_task = progress.add_task('Constructing edges...', total = len(df))
            for user_id, quoter_id in zip(
                has_quoter['user.id'].values,
                has_quoter['quoted_status.user.id'].values
            ):
                edges_to += [quoter_id]
                edges_from += [user_id]
                progress.update(graph_task, advance =1)

        nodes = list(set(edges_to) | set(edges_from))
        edges = list(zip(edges_from, edges_to))
        print(f'Constructed graph with {len(nodes):,} nodes and {len(edges_to):,} edges in {int(time.time() - start)} seconds')
        
        json.dump(
            nodes,
            open(data_config.graph_nodes_file, 'w')
        )
        json.dump(
            edges,
            open(data_config.graph_edges_file, 'w')
        )
        print(f'Saved node and edge information to {data_config.graph_dir}')
    else:
        start = time.time()
        nodes = json.load(open(data_config.graph_nodes_file))
        edges = json.load(open(data_config.graph_edges_file))
        print(f'loaded graph edges in {int(time.time() - start)} seconds')

    return nodes, edges


