import yaml
from pathlib import Path
import os

ES_KEYWORDS = yaml.safe_load(open(Path(__file__).parent/'yaml/keywords.yaml'))
ES_COLUMNS = yaml.safe_load(open(Path(__file__).parent/'yaml/columns.yaml'))
ES_QUERY = yaml.safe_load(open(Path(__file__).parent/'yaml/query.yaml'))

ES_HOSTNAME = os.environ.get('ES_HOSTNAME')
ES_INDEXNAME = os.environ.get('ES_INDEXNAME')
