import click
from crypto_twitter.data import (
    load_raw_data,
    build_graph,
    graph_stats,
)


@click.command()
@click.argument('operation')
def run(
    operation: str
):
    if operation == 'load_raw_data':
        load_raw_data()
    if operation == 'build_graph':
        build_graph()
    if operation == 'graph_stats':
        graph_stats()
    else:
        raise Exception('Unknown operation')

if __name__ == "__main__":
    run()
