from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent.parent
BASE_CONFIG_DIR = BASE_DIR / 'config'
load_dotenv(BASE_DIR / '.env')

if os.environ.get('ES_HOSTNAME'):
    ES_HOSTNAME = os.environ.get('ES_HOSTNAME')
else:
    raise Exception('Unable to determine ES_HOSTNAME')

if os.environ.get('DATA_DIR'):
    DATA_DIR = Path(str(os.environ.get('DATA_DIR')))
else:
    raise Exception('Unable to determine DATA_DIR')
# Figures
FIGS_DIR = BASE_DIR / 'figures'
FIGS_DIR.mkdir(exist_ok=True, parents=True)


