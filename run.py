import click

from crypto_chatter.graph import CryptoTwitterReplyGraph

@click.command()
@click.argument('operation')
@click.option('-d', '--dataset', type=str, default='twitter:blockchain-interoperability-attacks')
@click.option('-t', '--graph_type', type=str, default='reply')
def run(
    operation: str,
    dataset: str,
    graph_type: str,
):
    data_source, index_name = dataset.split(':')
    if data_source == 'twitter':
        if graph_type == 'reply': 
            graph = CryptoTwitterReplyGraph(index_name)
        else: 
            raise Exception('Unknown graph type')
    elif data_source == 'reddit':
        raise Exception('Reddit is not implemented')
    else:
        raise Exception('Unknown data source')

    if operation == 'graph_overview':
        graph.get_stats(display=True)
    elif operation == 'recompute_graph_overview':
        graph.get_stats(recompute=True, display=True)
    elif operation == 'export_components_gephi':
        graph.export_gephi_components()
    else:
        raise Exception('Unknown operation')

if __name__ == "__main__":
    run()
