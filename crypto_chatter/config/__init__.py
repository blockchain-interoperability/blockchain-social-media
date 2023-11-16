from .path import (
    BASE_DIR, 
    DATA_DIR,
)

from .presets import (
    ES_HOSTNAME,
    ES_TWITTER_KEYWORDS,
    ES_TWITTER_COLUMNS,
    ES_TWITTER_MAPPINGS,
    ES_TWITTER_QUERY,
    ES_REDDIT_COLUMNS,
    ES_REDDIT_QUERY,
    ES_REDDIT_MAPPINGS,
    # REDDIT_USERNAME,
    # REDDIT_PASSWORD,
    # REDDIT_CLIENT_ID,
    # REDDIT_CLIENT_SECRET,
)

from .crypto_chatter_data_config import CryptoChatterDataConfig
from .crypto_chatter_graph_config import CryptoChatterGraphConfig
from .default.data_config import load_default_data_config
from .default.graph_config import load_default_graph_config
