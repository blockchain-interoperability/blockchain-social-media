import pandas as pd
import json
import time

from crypto_twitter.utils import progress_bar
from crypto_twitter.config import GRAPH_DIR
from crypto_twitter.data import load_raw_data

def build_graph() -> tuple[list[int],list[tuple[int]]|list[list[int]]]:

    node_file = GRAPH_DIR / 'nodes.json'
    edge_file = GRAPH_DIR / 'edges.json'

    if not node_file.is_file() and not edge_file.is_file():
        df = load_raw_data()
        edges_to = []
        edges_from = []

        start = time.time()
        with progress_bar() as progress:
            graph_task = progress.add_task('Constructing edges...', total = len(df))
            for tweet_id, reply_id in df[['id','quoted_status.id']].values:
                if not pd.isna(reply_id):
                    edges_to += [reply_id]
                    edges_from += [tweet_id]
                progress.update(graph_task, advance =1)

        nodes = list(set(edges_to) | set(edges_from))
        edges = list(zip(edges_from, edges_to))
        print(f'Constructed graph with {len(nodes):,} nodes and {len(edges_to):,} edges in {int(time.time() - start)} seconds')
        
        json.dump(
            nodes,
            open(node_file, 'w')
        )
        json.dump(
            edges,
            open(edge_file, 'w')
        )
        print(f'Saved node and edge information to {GRAPH_DIR}')
    else:
        start = time.time()
        nodes = json.load(open(node_file))
        edges = json.load(open(edge_file))
        print(f'loaded graph in {int(time.time() - start)} seconds')

    return nodes, edges


