import click
from crypto_twitter.data import load_raw_data

@click.command()
@click.argument('operation')
def run(
    operation: str
):
    if operation == 'load_raw_data':
        load_raw_data()
    else:
        raise Exception('Unknown operation')

if __name__ == "__main__":
    run()
