from dataclasses import dataclass
from pathlib import Path

@dataclass
class CryptoChatterDataConfig:
    data_source: str
    raw_snapshot_dir: Path
    graph_dir: Path
    graph_stats_file: Path
    graph_edges_file: Path
    graph_nodes_file: Path
    graph_data_file: Path
    graph_gephi_file: Path
    graph_attributes: list[str]
    reddit_username: str | None = None
    reddit_password: str | None = None
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    es_index: str | None = None
    es_hostname: str | None = None
    es_query: dict | None = None
    es_columns: list | None = None
    es_mappings: dict | None = None
