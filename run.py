import click

from crypto_chatter.graph import CryptoTwitterGraph

@click.command()
@click.argument('operation')
@click.option('-d', '--dataset', type=str, default='twitter')
def run(
    operation: str,
    dataset: str,
):
    if dataset == 'twitter': 
        graph = CryptoTwitterGraph()
    elif dataset == 'reddit':
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
    else:
        raise Exception('Unknown operation')

if __name__ == "__main__":
    run()
