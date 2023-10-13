import pandas as pd

from crypto_twitter.config import GRAPH_DATA_FILE
from crypto_twitter.data import (
    load_graph_edges,
)
from .load_raw_data import load_raw_data

# Only load data that is used in the graph
def load_graph_data() -> pd.DataFrame:
    if not GRAPH_DATA_FILE.is_file():
        raw_df = load_raw_data()
        nodes, _ = load_graph_edges()

        graph_df = raw_df[raw_df['id'].isin(nodes)]
        graph_df.to_pickle(GRAPH_DATA_FILE)
    else:
        graph_df = pd.read_pickle(GRAPH_DATA_FILE)

    return graph_df
