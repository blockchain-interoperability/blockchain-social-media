import time
import pandas as pd

from crypto_chatter.config import CryptoChatterDataConfig
from .load_raw_data import load_raw_data

# Only load data that is used in the graph
def load_graph_data(
    nodes: list[int],
    data_config: CryptoChatterDataConfig,
) -> pd.DataFrame:
    if not data_config.graph_data_file.is_file():
        raw_df = load_raw_data(data_config)
        graph_df = raw_df[raw_df['id'].isin(nodes)]
        graph_df.to_pickle(data_config.graph_data_file)
        print('saved graph data to cache')
    else:
        start = time.time()
        graph_df = pd.read_pickle(data_config.graph_data_file)
        print(f'loaded cached graph data in {int(time.time() - start)} seconds')

    return graph_df
