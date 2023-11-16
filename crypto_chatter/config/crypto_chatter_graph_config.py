from typing import Literal
from pathlib import Path
from dataclasses import dataclass

@dataclass
class CryptoChatterGraphConfig:
    data_source: str
    graph_type: Literal['tweet','user']
    graph_dir: Path
    node_id_col: str
