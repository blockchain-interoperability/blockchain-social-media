from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent

from dotenv import load_dotenv
load_dotenv(BASE_DIR / '.env')
import os

if os.environ.get('DATA_DIR'):
    DATA_DIR = Path(str(os.environ.get('DATA_DIR')))
else:
    raise Exception('Unable to determine DATA_DIR')

RAW_SNAPSHOT_DIR = DATA_DIR / 'snapshots'
RAW_SNAPSHOT_DIR.mkdir(exist_ok=True, parents=True)
