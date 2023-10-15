import time
import pandas as pd

from crypto_chatter.config import CryptoChatterDataConfig
from .load_graph_edges import load_graph_edges
from .load_raw_data import load_raw_data

# Only load data that is used in the graph
def load_graph_data(data_config: CryptoChatterDataConfig) -> tuple[pd.DataFrame, list[int], list[list[int]]]:
    nodes, edges = load_graph_edges(data_config)
    if not data_config.graph_data_file.is_file():
        raw_df = load_raw_data(data_config)

        graph_df = raw_df[raw_df['id'].isin(nodes)]
        graph_df.to_pickle(data_config.graph_data_file)
        print('saved graph data to cache')
    else:
        start = time.time()
        graph_df = pd.read_pickle(data_config.graph_data_file)
        print(f'loaded cached graph in {int(time.time() - start)} seconds')

    return graph_df, nodes, edges
