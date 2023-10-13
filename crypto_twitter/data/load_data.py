from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd

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
    print(f'I should have {len(ES_COLUMNS)} columns')

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
    df[df['truncated']]['full_text'] = df[df['truncated']]['extended_tweet.full_text']

    df['quoted_status.truncated'] = df['quoted_status.truncated'].fillna(False)
    df['quoted_status.full_text'] = df['quoted_status.text']
    df[df['quoted_status.truncated']]['quoted_status.full_text'] = df[df['quoted_status.truncated']]['quoted_status.extended_tweet.full_text']

    return df

def load_raw_data() -> pd.DataFrame:
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
    print(f'scanning {doc_count:,} documents..')

    # if there are no snapshots or the last (sorted) snapshot does not match the doc count, start over
    if not RAW_SNAPSHOT_DIR.is_file():
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
            for c in cursor:
                results += [c]
                progress.update(scroll_task, advance = 1)

        print('turning into dataframe')
        df = prettify_elastic(results)

        df.to_json(RAW_SNAPSHOT_DIR, orient='records')
    else:
        df = pd.read_json(RAW_SNAPSHOT_DIR)

    return df

if __name__ == "__main__":
    load_raw_data()
