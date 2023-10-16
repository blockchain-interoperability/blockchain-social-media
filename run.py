import click

from crypto_chatter.graph import CryptoTwitterGraph

@click.command()
@click.argument('operation')
@click.option('-d', '--dataset', type=str, default='twitter:blockchain-interoperability-attacks')
def run(
    operation: str,
    dataset: str,
):
    data_source, index_name = dataset.split(':')
    if data_source == 'twitter': 
        graph = CryptoTwitterGraph(index_name)
    elif data_source == 'reddit':
        graph = None
    else:
        raise Exception('Unknown data source')
    if operation == 'build_graph':
        graph.build()
    elif operation == 'graph_overview':
        graph.build()
        graph.get_stats(display=True)
    elif operation == 'recompute_graph_overview':
        graph.build()
        graph.get_stats(recompute=True, display=True)
    elif operation == 'export_gephi':
        graph.build()
        graph.export_gephi()
    else:
        raise Exception('Unknown operation')

if __name__ == "__main__":
    run()
