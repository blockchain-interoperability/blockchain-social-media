import yaml
from pathlib import Path

KEYWORDS = yaml.safe_load(open(Path(__file__).parent/'yaml/keywords.yaml'))
COLUMNS = yaml.safe_load(open(Path(__file__).parent/'yaml/columns.yaml'))
ELASTIC_CONFIG = yaml.safe_load(open(Path(__file__).parent/'yaml/elastic.yaml'))

