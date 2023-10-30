import click

from crypto_chatter.graph import load_graph 
from crypto_chatter.config import load_default_data_config

@click.command()
@click.argument('operation')
@click.option('-d', '--dataset', type=str, default='twitter:blockchain-interoperability-attacks')
@click.option('-t', '--graph_type', type=str, default='tweet')
def run(
    operation: str,
    dataset: str,
    graph_type: str,
):
    data_config = load_default_data_config(dataset, graph_type)
    graph = load_graph(data_config, graph_type)

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
