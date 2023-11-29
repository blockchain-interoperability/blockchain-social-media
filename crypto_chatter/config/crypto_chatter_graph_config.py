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
        graph_kind: GraphKind,
    ):
        if graph_kind in typing.get_args(TwitterGraphKind):
            if data_config.data_source != "twitter":
                raise Exception("Twitter graph types are only supported for Twitter data sources")
            if graph_kind == "tweet-quote":
                self.edge_from_col = "id"
                self.edge_to_col = "quoted_status.id"
                self.is_directed = True
            elif graph_kind == "tweet-reply":
                self.edge_from_col = "id"
                self.edge_to_col = "in_reply_to_status_id"
                self.is_directed = True
            elif graph_kind == "user-quote":
                self.edge_from_col = "user.id"
                self.edge_to_col = "quoted_status.user.id"
                self.is_directed = True
            elif graph_kind == "user-reply":
                self.edge_from_col = "user.id"
                self.edge_to_col = "in_reply_to_user_id"
                self.is_directed = True
            else:
                raise NotImplementedError(f"{graph_kind} graph type is yet implemented!")
        else:
            raise NotImplementedError(f"{graph_kind} graph type is yet implemented!")

        self.graph_kind = graph_kind
        self.graph_dir = data_config.data_dir.parent / "graphs" / graph_kind
        self.graph_dir.mkdir(parents=True, exist_ok=True)
