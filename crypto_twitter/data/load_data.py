from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from pathlib import Path
import pandas as pd
import h5py

from crypto_twitter.utils.progress import progress_bar
from crypto_twitter.config.elastic import (
    ELASTIC_CONFIG,
    KEYWORDS,
)
from crypto_twitter.config.paths import SNAPSHOT_DIR

def prettify_elastic(results) -> pd.DataFrame:
    """Cleans up list of results into a dataframe while dropping some unncessary columns.

    Args:
        results (_type_): List of dataframes with a shared column.

    Returns:
        df (pd.DataFrame): Concatenated dataframe with indexes reset
    """
    df = pd.json_normalize(results)
    df = df.drop(columns=df.columns[~df.columns.str.contains('_source')])
    df.columns = df.columns.str.replace('_source.','').str.replace('.','_')

    # If trucated is False, that means text has the full text. If True, extended_tweet.full_text has the full text.
    
    df['truncated'] = df['truncated'].fillna(2)
    df['whole_text'] = df[
        [
            'extended_tweet_full_text',
            'text',     
            'truncated',
        ]
    ].apply(
        lambda row: row[0] if row[2] == True else row[1] if row[2] == False else '',
        axis=1
    )

    return df

def load_data() -> pd.DataFrame:
    """Grabs the specified fields from the specified index on Elasticsearch. Since results are expected to be larged, batched pickle files are generated

    Args:
        SNAPSHOT_DIR (str, optional): Path of directory that stores snapshots. Defaults to ''.
    Returns:
        df (pd.DataFrame): A concatenated dataframe containing all the specified columns.
    """

    es = Elasticsearch(
        hosts=[ELASTIC_CONFIG['hostname']],
        verify_certs=False,
    )    
    # we will add a query to only grab the ones that contain at least one keyword (or partially, if keyword was space separated)
    mainquery = ELASTIC_CONFIG['mainquery']
    mainquery['query']['bool']['must'] = {
        "simple_query_string": {
            "query": ' '.join(KEYWORDS),
            "fields": [
                "text",
                "extended_tweet.full_text"
            ],
        }
    }
    doc_count = es.count(
        index=[ELASTIC_CONFIG['index_name']],
        body=mainquery,
        request_timeout = 120,
    )['count']



    # if there are no snapshots or the last (sorted) snapshot does not match the doc count, start over
    if not SNAPSHOT_DIR.is_file():
        cursor = scan(
            es,
            index=ELASTIC_CONFIG['index_name'],
            query = {**mainquery, '_source':ELASTIC_CONFIG['fields']},
            size=10000,
            request_timeout = 120,
        )
        results = []
        with progress_bar() as progress:
            scroll_task = progress.add_task(description='scrolling index.. ', total=doc_count)
            for c in cursor:
                results += [c]
                progress.update(scroll_task, advance = 1)
                break

        print('turning into dataframe')
        df = prettify_elastic(results)

        print('loaded dataframe')
        print('columns:')
        for c in df.columns:
            print(c)

        # with h5py.File(str(SNAPSHOT_DIR), 'w') as f:
        #     dataset = f.create_dataset('')

        
    # we have everything, just need to concat the dataframes
    else:
        print('loading from cache...')
        df = pd.DataFrame([
            pd.read_pickle(f)
            for f in SNAPSHOT_DIR.glob('*.pkl')
        ])

    return df

