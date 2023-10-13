from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
import time

from crypto_twitter.utils.progress import progress_bar
from crypto_twitter.config import (
    ES_QUERY,
    ES_COLUMNS,
    ES_HOSTNAME,
    ES_INDEXNAME,
    ES_KEYWORDS,
    RAW_SNAPSHOT_DIR,
)

def prettify_elastic(results) -> pd.DataFrame:
    """Cleans up list of results into a dataframe while dropping some unncessary columns.

    Args:
        results (_type_): List of dataframes with a shared column.

    Returns:
        df (pd.DataFrame): Concatenated dataframe with indexes reset
    """

    df = pd.json_normalize(results)
    df = df.drop(columns=df.columns[~df.columns.str.contains('_source')])
    df.columns = df.columns.str.replace('_source.','')
    df = df.reindex(
        columns = df.columns.union(
            [
                'full_text', 
                'quoted_status.full_text',
                'truncated'
                'quoted_status.truncated',
            ] + ES_COLUMNS
        )
    )

    # If trucated is False, that means text has the full text. If True, extended_tweet.full_text has the full text.
    df['truncated'] = df['truncated'].fillna(False)
    df['full_text'] = df['text']
    df.loc[df['truncated'],'full_text'] = df[df['truncated']]['extended_tweet.full_text']

    df['quoted_status.truncated'] = df['quoted_status.truncated'].fillna(False)
    df['quoted_status.full_text'] = df['quoted_status.text']
    df.loc[df['quoted_status.truncated'],'quoted_status.full_text'] = df[df['quoted_status.truncated']]['quoted_status.extended_tweet.full_text']

    return df

def load_raw_data() -> pd.DataFrame:
    """Grabs the specified fields from the specified index on Elasticsearch. Since results are expected to be larged, batched pickle files are generated

    Args:
        SNAPSHOT_DIR (str, optional): Path of directory that stores snapshots. Defaults to ''.
    Returns:
        df (pd.DataFrame): A concatenated dataframe containing all the specified columns.
    """
    
    if not (RAW_SNAPSHOT_DIR / 'final.pkl').is_file():
        chunk_size = 100000
        es = Elasticsearch(
            hosts=[ES_HOSTNAME],
            verify_certs=False,
        )    
        # we will add a query to only grab the ones that contain at least one keyword (or partially, if keyword was space separated)
        ES_QUERY['query']['bool']['must'] = {
            "simple_query_string": {
                "query": ' '.join(ES_KEYWORDS),
                "fields": [
                    "text",
                    "extended_tweet.full_text"
                ],
            }
        }
        doc_count = es.count(
            index=[ES_INDEXNAME],
            body=ES_QUERY,
            request_timeout = 120,
        )['count']
        print(f'scanning {doc_count:,} documents..')

        cursor = scan(
            es,
            index=ES_INDEXNAME,
            query = {**ES_QUERY, '_source': ES_COLUMNS},
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
                    df = prettify_elastic(results)
                    df.to_pickle(
                        RAW_SNAPSHOT_DIR / f'{len(dataframes):010d}.pkl', 
                    )
                    del results[:]
                    dataframes += [df]
                progress.update(scroll_task, advance = 1)

        df = prettify_elastic(results)
        del results[:]
        df.to_pickle(
            RAW_SNAPSHOT_DIR / f'final.pkl',
        )
        dataframes += [df]

        print(f'we saved {(len(dataframes) -1) * chunk_size + len(results):,} rows in {len(dataframes)} chunks in {int(time.time()-start)} seconds')
        df = pd.concat(dataframes)

    else:
        start = time.time()
        dataframes = []
        cache_files = sorted(RAW_SNAPSHOT_DIR.glob('*.pkl'))
        with progress_bar() as progress:
            load_task = progress.add_task(description='loading cache...', total=len(cache_files))
            for file in cache_files:
                dataframes += [pd.read_pickle(file)]
                progress.update(load_task, advance=1)
        df = pd.concat(dataframes)
        print(f'Loaded cache in {int(time.time()-start)} seconds')

    return df

if __name__ == "__main__":
    load_raw_data()
