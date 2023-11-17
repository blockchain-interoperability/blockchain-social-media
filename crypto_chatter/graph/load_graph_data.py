import pandas as pd
import json
import time

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig
from crypto_chatter.utils.types import (
    NodeList, 
    EdgeList,
    NodeToIndexMapping,
)

def load_graph_data(
    data_config: CryptoChatterDataConfig,
    graph_config: CryptoChatterGraphConfig,
    columns: list[str]|None = None,
) -> tuple[CryptoChatterData, NodeList, EdgeList, NodeToIndexMapping]:
    graph_config.graph_dir.mkdir(parents=True, exist_ok=True)
    graph_nodes_file = graph_config.graph_dir / "nodes.json"
    graph_edges_file = graph_config.graph_dir / "edges.json"
    graph_mapping_file = graph_config.graph_dir / "node_to_index.json"

    cols_to_load = []
    if graph_config.is_directed:
        cols_to_load += [graph_config.edge_to_col, graph_config.edge_from_col]
    else:
        raise NotImplementedError("only bidirectional graphs are supported")

    if columns is not None:
        cols_to_load += columns

    data = CryptoChatterData(
        data_config=data_config,
        cols_to_load=cols_to_load,
    )

    if (
        not graph_nodes_file.is_file() 
        or not graph_edges_file.is_file()
        or not graph_mapping_file.is_file()
    ):
        start = time.time()
        if graph_config.is_directed:
            has_outgoing: pd.DataFrame = data[~data[graph_config.edge_to_col].isna()]
            edges_from = has_outgoing[graph_config.edge_from_col].astype(int).tolist()
            edges_to = has_outgoing[graph_config.edge_to_col].astype(int).tolist()
            node_to_index = dict(zip(
                edges_from,
                has_outgoing.index 
            ))
            for node in edges_to:
                if node not in node_to_index:
                    node_to_index[node] = -1

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
        json.dump(
            node_to_index,
            open(graph_mapping_file, "w")
        )
        
        print(f"Constructed graph with {len(nodes):,} nodes and {len(edges_to):,} edges in {int(time.time() - start)} seconds")
        print(f"Saved node and edge information to {graph_config.graph_dir}")

    else:
        start = time.time()
        nodes = json.load(open(graph_nodes_file))
        edges = json.load(open(graph_edges_file))
        node_to_index = json.load(open(graph_mapping_file))
        node_to_index = {int(k): v for k,v in node_to_index.items()}
        print(f"loaded graph edges in {int(time.time() - start)} seconds")

    return data, nodes, edges, node_to_index
