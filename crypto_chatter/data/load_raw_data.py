from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
import time

from crypto_chatter.utils.progress import progress_bar
from crypto_chatter.config import CryptoChatterDataConfig

from .prettify_elastic_twitter import prettify_elastic_twitter

def prettify_elastic(results:list[dict], data_config:CryptoChatterDataConfig) -> pd.DataFrame:
    if data_config.data_source == 'twitter':
        df = prettify_elastic_twitter(results, data_config)
    elif data_config.data_source == 'reddit': 
        raise NotImplementedError('Reddit parsing is not yet implemented!')

    return df

def load_raw_data(data_config: CryptoChatterDataConfig) -> pd.DataFrame:
    """Grabs the specified fields from the specified index on Elasticsearch. Since results are expected to be larged, batched pickle files are generated

    Args:
        SNAPSHOT_DIR (str, optional): Path of directory that stores snapshots. Defaults to ''.
    Returns:
        df (pd.DataFrame): A concatenated dataframe containing all the specified columns.
    """
    
    if not (data_config.raw_snapshot_dir / 'final.pkl').is_file():
        chunk_size = 100000
        es = Elasticsearch(
            hosts=[data_config.es_hostname],
            verify_certs=False,
        )

        # we will add a query to only grab the ones that contain at least one keyword (or partially, if keyword was space separated)
        doc_count = es.count(
            index=[data_config.es_index],
            body=data_config.es_query,
            request_timeout = 120,
        )['count']
        print(f'scanning {doc_count:,} documents..')

        cursor = scan(
            es,
            index=data_config.es_index,
            query = {**data_config.es_query, '_source': data_config.es_columns},
            request_timeout = 120,
        )

        dataframes = []
        results = []
        start = time.time()
        with progress_bar() as progress:
            scroll_task = progress.add_task(description='scrolling index.. ', total=doc_count)
            for c in cursor:
                results += [c]
                if len(results) == chunk_size:
                    df = prettify_elastic(results, data_config)
                    df.to_pickle(
                        data_config.raw_snapshot_dir / f'{len(dataframes):010d}.pkl', 
                    )
                    del results[:]
                    dataframes += [df]
                progress.update(scroll_task, advance = 1)

        df = prettify_elastic(results, data_config)
        del results[:]
        df.to_pickle(
            data_config.raw_snapshot_dir / 'final.pkl',
        )
        dataframes += [df]

        print(f'we saved {(len(dataframes) -1) * chunk_size + len(results):,} rows in {len(dataframes)} chunks in {int(time.time()-start)} seconds')
        df = pd.concat(dataframes)

    else:
        start = time.time()
        dataframes = []
        cache_files = sorted(data_config.raw_snapshot_dir.glob('*.pkl'))
        with progress_bar() as progress:
            load_task = progress.add_task(description='loading cache...', total=len(cache_files))
            for file in cache_files:
                dataframes += [pd.read_pickle(file)]
                progress.update(load_task, advance=1)
        df = pd.concat(dataframes)
        print(f'Loaded cache in {int(time.time()-start)} seconds')

    return df
