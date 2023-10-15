from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent

from dotenv import load_dotenv
load_dotenv(BASE_DIR / '.env')
import os

if os.environ.get('DATA_DIR'):
    DATA_DIR = Path(str(os.environ.get('DATA_DIR')))
else:
    raise Exception('Unable to determine DATA_DIR')

# Figures
FIGS_DIR = BASE_DIR / 'figures'
FIGS_DIR.mkdir(exist_ok=True, parents=True)


