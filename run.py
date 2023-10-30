from typing import Literal
import click

from crypto_chatter.graph import (
    CryptoGraph,
    CryptoTwitterTweetGraph,
) 
from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.config.default import BlockchainAttackTwitterConfig

def load_data_config(dataset:str) -> CryptoChatterDataConfig:
    if dataset == 'twitter:blockchain-interoperability-attacks':
        data_config = BlockchainAttackTwitterConfig()
    else:
        raise Exception('Unknown data source')
    return data_config

def load_twitter_graph(
    data_config: CryptoChatterDataConfig,
    graph_type: Literal['tweet', 'user'],
) -> CryptoGraph:
    if graph_type == 'tweet':
        graph = CryptoTwitterTweetGraph(data_config=data_config)
    elif graph_type == 'user':
        raise NotImplementedError('User graph is not implemented')
    else:
        raise Exception('Unknown graph type')
    return graph


@click.command()
@click.argument('operation')
@click.option('-d', '--dataset', type=str, default='twitter:blockchain-interoperability-attacks')
@click.option('-t', '--graph_type', type=str, default='tweet')
def run(
    operation: str,
    dataset: str,
    graph_type: str,
):
    data_config = load_data_config(dataset)
    if 'twitter' in dataset:
        graph = load_twitter_graph(data_config, graph_type)
    else:
        raise NotImplementedError('Other data sources are not implemented')

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
