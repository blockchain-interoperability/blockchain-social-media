import click
import json
from crypto_twitter.data import (
    load_raw_data,
    load_graph_edges,
)
from crypto_twitter.graph import (
    get_graph_overview,
)


@click.command()
@click.argument('operation')
def run(
    operation: str
):
    if operation == 'load_raw_data':
        load_raw_data()
    elif operation == 'load_graph_edges':
        load_graph_edges()
    elif operation == 'view_graph_overview':
        overview = get_graph_overview()
        print(json.dumps(overview, indent=2))
    elif operation == 'recompute_graph_overview':
        overview = get_graph_overview(recompute=True)
        print(json.dumps(overview, indent=2))
    else:
        raise Exception('Unknown operation')

if __name__ == "__main__":
    run()
