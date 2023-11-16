import click

from crypto_chatter.config import load_default_data_config
from crypto_chatter.data import CryptoChatterData

@click.command()
@click.option('-d', '--dataset', type=str, default='twitter:blockchain-interoperability-attacks')

def run(
    dataset: str,
):
    data_config = load_default_data_config(dataset)
    data = CryptoChatterData(data_config)

if __name__ == "__main__":
    run()
