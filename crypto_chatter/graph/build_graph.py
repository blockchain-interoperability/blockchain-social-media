import json
import time
from rich.progress import Progress

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.config import CryptoChatterGraphConfig
from crypto_chatter.utils.types import (
    NodeList, 
    NodeToIdMap,
    EdgeList,
)

def build_graph(
    data: CryptoChatterData,
    graph_config: CryptoChatterGraphConfig,
    progress: Progress | None = None,
) -> tuple[NodeList, NodeToIdMap, EdgeList]:
    graph_config.graph_dir.mkdir(parents=True, exist_ok=True)
    graph_nodes_file = graph_config.graph_dir / "nodes.json"
    graph_node_to_ids_file = graph_config.graph_dir / "node_to_ids.json"
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
        or not graph_node_to_ids_file.is_file()
    ):
        start = time.time()
        if graph_config.is_directed:

            # need to filter by nan first b/c we can't convert to int
            has_text = data[data.data_config.text_col].notna()
            from_edge_valid = data[graph_config.edge_from_col].notna()
            to_edge_valid = data[graph_config.edge_to_col].notna()

            if 'tweet' in graph_config.graph_kind:
                is_valid = has_text&from_edge_valid&to_edge_valid
            elif 'user' in graph_config.graph_kind:
                is_valid = from_edge_valid&to_edge_valid
            else:
                raise NotImplementedError(f"{graph_config.graph_kind} graph type is yet implemented!")

            valid = data[is_valid]

            all_edges_to = valid[graph_config.edge_to_col].astype(int)
            all_edges_from = valid[graph_config.edge_from_col].astype(int)

            # we know that all nodes from are in the data, but not all nodes to may be present.
            nodes_in_ids = all_edges_to.isin(all_edges_from)

            edges_from = all_edges_to[nodes_in_ids]
            edges_to = all_edges_from[nodes_in_ids]

            nodes = list(set(edges_to) | set(edges_from))
            edges = list(set(zip(edges_from, edges_to)))

            if progress is not None:
                node_id_task = progress.add_task(
                    "mapping node to id",
                    total=int(data[graph_config.edge_from_col].nunique())
                )
            node_to_ids = {}

            for val, subset in data.df.groupby(graph_config.edge_from_col):
                node_to_ids[int(val)] = subset[data.data_config.id_col].values.tolist()

                if progress is not None:
                    progress.advance(node_id_task)

            if progress is not None:
                progress.remove_task(node_id_task)

        else:
            raise NotImplementedError("only bidirectional graphs are supported")

        json.dump(
            nodes,
            open(graph_nodes_file, "w")
        )
        json.dump(
            node_to_ids,
            open(graph_node_to_ids_file, "w")
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
        node_to_ids = json.load(open(graph_node_to_ids_file))
        node_to_ids = {int(k): v for k, v in node_to_ids.items()}
        edges = json.load(open(graph_edges_file))
        print(f"loaded graph edges in {time.time() - start:.2f} seconds")

    return nodes, node_to_ids, edges
