import json

from crypto_chatter.config import load_default_data_config, load_default_graph_config
from crypto_chatter.config.path import BASE_DIR
from crypto_chatter.data import CryptoChatterData

dataset = 'twitter:blockchain-interoperability-attacks'
graph_type = 'tweet'
data_config = load_default_data_config(dataset)
graph_config = load_default_graph_config(dataset, graph_type)

init_clusters = json.load(open(data_config.data_dir / 'kmeans_init_clusters.json'))

old_kmeans_dir = BASE_DIR / 'old/analysis-data/kmeans_clusters_resampled'
resample_clusters = {
    i: json.load(open(old_kmeans_dir / f'{i}_ids.json'))
    for i in range(6)
}

data = CryptoChatterData(data_config)
