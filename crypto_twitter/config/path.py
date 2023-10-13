from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent

from dotenv import load_dotenv
load_dotenv(BASE_DIR / '.env')
import os

if os.environ.get('DATA_DIR'):
    DATA_DIR = Path(str(os.environ.get('DATA_DIR')))
else:
    raise Exception('Unable to determine DATA_DIR')

# Snapshot from elasticsearch
RAW_SNAPSHOT_DIR = DATA_DIR / 'snapshots'
RAW_SNAPSHOT_DIR.mkdir(exist_ok=True, parents=True)

# Graph related stuff
GRAPH_DIR = DATA_DIR / 'graph'
GRAPH_DIR.mkdir(exist_ok=True, parents=True)

GRAPH_STATS_FILE = GRAPH_DIR / 'stats.json'
GRAPH_EDGES_FILE = GRAPH_DIR / 'edges.json'
GRAPH_NODES_FILE = GRAPH_DIR / 'nodes.json'
GRAPH_DATA_FILE = GRAPH_DIR / 'graph_df.pkl'

# Figures
FIGS_DIR = BASE_DIR / 'figures'
FIGS_DIR.mkdir(exist_ok=True, parents=True)


