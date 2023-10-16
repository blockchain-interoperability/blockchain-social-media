'''
Sets up elasticsearch index
'''

from elasticsearch import Elasticsearch

from crypto_chatter.config import CryptoChatterDataConfig, ES_TWITTER_MAPPINGS, ES_REDDIT_MAPPINGS

def verify_or_setup_index(
    es: Elasticsearch, 
    data_config: CryptoChatterDataConfig
) -> None:    
    #Do nothing if the index exists
    if es.indices.exists(data_config.es_index):
        print(f'Index \'{data_config.es_index}\' found.') 
        return
    
    #Create if the index does not exist
    es.indices.create(data_config.es_index, {
        'mappings': data_config.es_mappings
    })

    return 
