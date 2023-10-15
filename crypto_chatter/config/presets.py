import yaml
from pathlib import Path
import os

ES_TWITTER_KEYWORDS = yaml.safe_load(open(Path(__file__).parent/'yaml/twitter/keywords.yaml'))
ES_TWITTER_COLUMNS = yaml.safe_load(open(Path(__file__).parent/'yaml/twitter/columns.yaml'))
ES_TWITTER_QUERY = yaml.safe_load(open(Path(__file__).parent/'yaml/twitter/query.yaml'))

if os.environ.get('ES_HOSTNAME'):
    ES_HOSTNAME = os.environ.get('ES_HOSTNAME')
else:
    raise Exception('Unable to determine DATA_DIR')
if os.environ.get('ES_TWITTER_INDEXNAME'):
    ES_TWITTER_INDEXNAME = os.environ.get('ES_TWITTER_INDEXNAME')
else:
    raise Exception('Unable to determine DATA_DIR')
