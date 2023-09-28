from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from pathlib import Path
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import h5py
import click
import json

from crypto_twitter.utils.progress import progress_bar
from crypto_twitter.config import (
    ES_QUERY,
    ES_COLUMNS,
    ES_HOSTNAME,
    ES_INDEXNAME,
    ES_KEYWORDS,
    SNAPSHOT_DIR,
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

    # If trucated is False, that means text has the full text. If True, extended_tweet.full_text has the full text.
    df['truncated'] = df['truncated'].fillna(2)
    df['full_text'] = df[
        [
            'extended_tweet.full_text',
            'text',     
            'truncated',
        ]
    ].apply(
        lambda row: (
            row[0] if row[2] == True 
            else row[1] if row[2] == False 
            else ''
        ),
        axis=1
    )

    df['quoted_status.full_text'] = df[
        [
            'quoted_status.extended_tweet.full_text',
            'quoted_status.text',     
            'quoted_status.truncated',
        ]
    ].apply(
        lambda row: (
            row[0] if row[2] == True 
            else row[1] if row[2] == False 
            else ''
        ),
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

    # if there are no snapshots or the last (sorted) snapshot does not match the doc count, start over
    # if not SNAPSHOT_DIR.is_file():
    cursor = scan(
        es,
        index=ES_INDEXNAME,
        query = {**ES_QUERY, '_source': ES_COLUMNS},
        size=10000,
        request_timeout = 120,
    )
    results = []
    with progress_bar() as progress:
        scroll_task = progress.add_task(description='scrolling index.. ', total=doc_count)
        counter = 0
        for c in cursor:
            results += [c]
            progress.update(scroll_task, advance = 1)
            if counter == 10: break
            counter += 1

    print('turning into dataframe')
    df = prettify_elastic(results)

    print('loaded dataframe')

    str_type = h5py.special_dtype(vlen=str)
    
    # with h5py.File(str(SNAPSHOT_DIR), 'w') as f:
    #     for col in ['full_text'] + ES_COLUMNS:
    #         print(col, df[col].dtype)
    #         if is_string_dtype(df[col].dtype):
    #             f.create_dataset(col, data=df[col].values, dtype=str_type)
    #         else:
    #             f.create_dataset(col, data=df[col].values)
    
    # we have everything, just need to concat the dataframes
    # else:
    #     print('loading from cache...')
    #     with h5py.File(str(SNAPSHOT_DIR), 'r') as f:
    #         df = pd.concat(
    #             [
    #                 pd.Series(f[c][:], name = c)
    #                 for col in ['full_text'] + ES_COLUMNS
    #             ],
    #             axis = 1
    #         )

    return df

if __name__ == "__main__":
    load_data()
