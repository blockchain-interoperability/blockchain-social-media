from crypto_chatter.config import (
    ES_TWITTER_QUERY,
    ES_HOSTNAME,
    ES_TWITTER_COLUMNS,
    ES_TWITTER_MAPPINGS,
    ES_TWITTER_KEYWORDS,
    DATA_DIR,
    CryptoChatterDataConfig,
)
es_query = ES_TWITTER_QUERY
es_query['query']['bool']['must'] = {
    "simple_query_string": {
        "query": ' '.join(ES_TWITTER_KEYWORDS),
        "fields": [
            "text",
            "extended_tweet.full_text"
        ],
    }
}
index_name = 'blockchain-interoperability-attacks'
data_source = 'twitter'

class BlockchainAttackTwitterTweetGraphConfig(CryptoChatterDataConfig):
    es_hostname = ES_HOSTNAME
    es_index = index_name
    es_columns = ES_TWITTER_COLUMNS
    es_mappings = ES_TWITTER_MAPPINGS
    es_query = es_query
    data_source = data_source
    node_id_col ='id'
    text_col ='full_text'
    raw_snapshot_dir = DATA_DIR / f'twitter/{index_name}/snapshots'
    graph_type = 'tweet'
    graph_dir = DATA_DIR / f'twitter/{index_name}/tweet-graph'