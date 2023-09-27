import yaml
from pathlib import Path

KEYWORDS = yaml.safe_load(open(Path(__file__)/'yaml/keywords.yaml'))
COLUMNS = yaml.safe_load(open(Path(__file__)/'yaml/columns.yaml'))
ELASTIC_CONFIG = yaml.safe_load(open(Path(__file__)/'yaml/elastic_config.yaml'))

