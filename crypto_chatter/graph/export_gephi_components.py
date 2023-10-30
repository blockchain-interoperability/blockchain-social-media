import networkx as nx

from crypto_chatter.utils import progress_bar

from .crypto_graph import CryptoGraph

def export_gephi_components(
    graph: CryptoGraph
) -> None:
    graph.load_components()
    with progress_bar() as progress:
        save_task = progress.add_task('exporting components to gephi..', total=len(graph.components))
        for i,c in enumerate(graph.components):
            subgraph = graph.G.subgraph(c)
            for col in graph.data.columns:
                nx.set_node_attributes(
                    subgraph,
                    values = dict(zip(
                        graph.data[graph.data_config.node_id_col].values.tolist(), 
                        graph.data[col].values.tolist()
                    )),
                    name = col,
                )
            nx.write_gexf(subgraph, graph.data_config.graph_gephi_dir / f'{i:06d}.gexf')
            progress.advance(save_task)
