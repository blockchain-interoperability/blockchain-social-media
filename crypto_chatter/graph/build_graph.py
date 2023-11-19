import pandas as pd
import json
import time

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.config import CryptoChatterGraphConfig
from crypto_chatter.utils.types import (
    NodeList, 
    EdgeList,
)

def build_graph(
    data: CryptoChatterData,
    graph_config: CryptoChatterGraphConfig,
) -> tuple[NodeList, EdgeList]:
    graph_config.graph_dir.mkdir(parents=True, exist_ok=True)
    graph_nodes_file = graph_config.graph_dir / "nodes.json"
    graph_edges_file = graph_config.graph_dir / "edges.json"

    cols_to_load = []
    if graph_config.is_directed:
        cols_to_load += [graph_config.edge_to_col, graph_config.edge_from_col]
    else:
        raise NotImplementedError("only bidirectional graphs are supported")

    data.load(cols_to_load)

    if (
        not graph_nodes_file.is_file() 
        or not graph_edges_file.is_file()
    ):
        start = time.time()
        if graph_config.is_directed:
            has_outgoing: pd.DataFrame = data[(
                ~data[graph_config.edge_to_col].isna()
                & data[graph_config.edge_to_col].isin(
                    data.ids
                )
                & data[graph_config.edge_from_col].isin(
                    data.ids
                )
            )]

            # need to filter by nan first b/c we can't convert to int
            has_text = data[data.data_config.text_col].notna()
            from_edge_valid = data[graph_config.edge_from_col].notna()
            to_edge_valid = data[graph_config.edge_to_col].notna()
            has_valid_values = data[(has_text & from_edge_valid & to_edge_valid)]

            all_to_edge = has_valid_values[graph_config.edge_to_col].astype(int)
            all_from_edge = has_valid_values[graph_config.edge_from_col].astype(int)

            nodes_in_ids = (all_to_edge.isin(data.ids) & all_from_edge.isin(data.ids))

            edges_from = all_to_edge[nodes_in_ids]
            edges_to = all_from_edge[nodes_in_ids]

            nodes = list(set(edges_to) | set(edges_from))
            edges = list(zip(edges_from, edges_to))
        else:
            raise NotImplementedError("only bidirectional graphs are supported")

        json.dump(
            nodes,
            open(graph_nodes_file, "w")
        )
        json.dump(
            edges,
            open(graph_edges_file, "w")
        )
        
        print(f"Constructed graph with {len(nodes):,} nodes and {len(edges_to):,} edges in {time.time() - start:.2f} seconds")
        print(f"Saved node and edge information to {graph_config.graph_dir}")

    else:
        start = time.time()
        nodes = json.load(open(graph_nodes_file))
        edges = json.load(open(graph_edges_file))
        print(f"loaded graph edges in {time.time() - start:.2f} seconds")

    return nodes, edges
