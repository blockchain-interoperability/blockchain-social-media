import typing
from pathlib import Path

from crypto_chatter.utils.types import (
    GraphKind,
    TwitterGraphKind
)

from .crypto_chatter_data_config import CryptoChatterDataConfig

class CryptoChatterGraphConfig:
    graph_type: GraphKind
    graph_dir: Path
    edge_from_col: str
    edge_to_col: str
    is_directed: bool

    def __init__(
        self,
        data_config: CryptoChatterDataConfig,
        graph_type: GraphKind,
    ):
        if graph_type in typing.get_args(TwitterGraphKind):
            if data_config.data_source != 'twitter':
                raise Exception('Twitter graph types are only supported for Twitter data sources')
            if graph_type == 'tweet':
                self.edge_from_col = 'id'
                self.edge_to_col = 'quoted_status.id'
                self.is_directed = True
            else:
                raise NotImplementedError(f'{graph_type} graph type is yet implemented!')
        else:
            raise NotImplementedError(f'{graph_type} graph type is yet implemented!')

        self.graph_type = graph_type
        self.graph_dir = data_config.data_dir.parent / 'graphs' / graph_type
        self.graph_dir.mkdir(parents=True, exist_ok=True)