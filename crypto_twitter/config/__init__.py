from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

from .path import BASE_DIR
load_dotenv(BASE_DIR / '.env')

DATA_DIR = Path(os.environ.get('DATA_DIR'))
SNAPSHOT_DIR = DATA_DIR / 'snapshot.h5'

from .elastic import (
    ES_KEYWORDS,
    ES_COLUMNS,
    ES_QUERY,
    ES_HOSTNAME,
    ES_INDEXNAME
)
