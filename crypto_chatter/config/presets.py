import yaml
from pathlib import Path
import os

ES_TWITTER_KEYWORDS = yaml.safe_load(open(Path(__file__).parent/'yaml/twitter/keywords.yaml'))
ES_TWITTER_COLUMNS = yaml.safe_load(open(Path(__file__).parent/'yaml/twitter/columns.yaml'))
ES_TWITTER_QUERY = yaml.safe_load(open(Path(__file__).parent/'yaml/twitter/query.yaml'))
ES_TWITTER_MAPPINGS = yaml.safe_load(open(Path(__file__).parent/'yaml/twitter/mappings.yaml'))

ES_REDDIT_COLUMNS = yaml.safe_load(open(Path(__file__).parent/'yaml/reddit/columns.yaml'))
ES_REDDIT_QUERY = yaml.safe_load(open(Path(__file__).parent/'yaml/reddit/query.yaml'))
ES_REDDIT_MAPPINGS = yaml.safe_load(open(Path(__file__).parent/'yaml/reddit/mappings.yaml'))

if os.environ.get('ES_HOSTNAME'):
    ES_HOSTNAME = os.environ.get('ES_HOSTNAME')
else:
    raise Exception('Unable to determine ES_HOSTNAME')
    
if os.environ.get('REDDIT_USERNAME'):
    REDDIT_USERNAME = os.environ.get('REDDIT_USERNAME')
else:
    raise Exception('Unable to determine REDDIT_USERNAME')

if os.environ.get('REDDIT_PASSWORD'):
    REDDIT_PASSWORD = os.environ.get('REDDIT_PASSWORD')
else:
    raise Exception('Unable to determine REDDIT_PASSWORD')

if os.environ.get('REDDIT_CLIENT_ID'):
    REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
else:
    raise Exception('Unable to determine REDDIT_CLIENT_ID')

if os.environ.get('REDDIT_CLIENT_SECRET'):
    REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
else:
    raise Exception('Unable to determine REDDIT_CLIENT_SECRET')

