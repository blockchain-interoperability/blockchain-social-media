from pathlib import Path
import yaml

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path(config['data_dir'])
SNAPTSHOT_DIR = DATA_DIR / 'snapshot.h5'
