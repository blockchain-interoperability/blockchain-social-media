import yaml
from pathlib import Path

from .path import (
    BASE_CONFIG_DIR,
    DATA_DIR,
    ES_HOSTNAME,
)

class CryptoChatterDataConfig:
    data_source: str
    raw_snapshot_dir: Path
    data_dir: Path
    reddit_username: str
    reddit_password: str
    reddit_client_id: str
    reddit_client_secret: str
    es_index: str
    es_hostname: str
    es_query: dict
    es_columns: list
    es_mappings: dict
    id_col: str
    text_col: str = 'full_text'

    def __init__(
        self,
        config_name: str
    ) -> None:
        config_dir = BASE_CONFIG_DIR / config_name
        data_source, index_name = config_name.split(':')
        columns = yaml.safe_load(open(config_dir/'columns.yaml'))
        query = yaml.safe_load(open(config_dir/'query.yaml'))
        mappings = yaml.safe_load(open(config_dir/'mappings.yaml'))

        if (config_dir/'keywords.yaml').is_file():
            keywords = yaml.safe_load(open(config_dir/'keywords.yaml'))
            query['query']['bool']['must'] = {
                "simple_query_string": {
                    "query": ' '.join(keywords),
                    "fields": [
                        "text",
                        "extended_tweet.full_text"
                    ],
                }
            }

        self.data_source = data_source
        self.es_index = index_name
        self.es_query = query
        self.es_hostname = ES_HOSTNAME
        self.es_columns = columns
        self.es_mappings = mappings

        self.data_dir = DATA_DIR / f'{data_source}/{index_name}/data'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.raw_snapshot_dir = DATA_DIR / f'{data_source}/{index_name}/snapshots'
        self.raw_snapshot_dir.mkdir(parents=True, exist_ok=True)

        if data_source == 'twitter':
            self.id_col = 'id'
