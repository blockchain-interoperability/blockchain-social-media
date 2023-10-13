import yaml
from pathlib import Path
import os

ES_KEYWORDS = yaml.safe_load(open(Path(__file__).parent/'yaml/keywords.yaml'))
ES_COLUMNS = yaml.safe_load(open(Path(__file__).parent/'yaml/columns.yaml'))
ES_QUERY = yaml.safe_load(open(Path(__file__).parent/'yaml/query.yaml'))

if os.environ.get('ES_HOSTNAME'):
    ES_HOSTNAME = os.environ.get('ES_HOSTNAME')
else:
    raise Exception('Unable to determine DATA_DIR')
if os.environ.get('ES_INDEXNAME'):
    ES_INDEXNAME = os.environ.get('ES_INDEXNAME')
else:
    raise Exception('Unable to determine DATA_DIR')
