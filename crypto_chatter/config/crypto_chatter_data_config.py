from dataclasses import dataclass
from pathlib import Path

@dataclass
class CryptoChatterDataConfig:
    es_index: str
    es_hostname: str
    es_query: dict
    es_columns: list
    data_source: str
    data_dir: Path
    raw_snapshot_dir: Path
    graph_dir: Path
    graph_stats_file: Path
    graph_edges_file: Path
    graph_nodes_file: Path
    graph_data_file: Path
