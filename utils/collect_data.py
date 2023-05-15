from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import json

# print()
# load the keywords for first round filtering
keywords = json.load(open(Path(__file__).parent.parent/'keywords.json'))
elastic_config = json.load(open(Path(__file__).parent.parent/'elastic_config.json'))


def has_keyword(text):
    return any(
        k.lower() in text.lower()
        for k in keywords
    )


def prettify_elastic(results):
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

def cache_index(snapshot_path,**kwargs):
    """Grabs the specified fields from the specified index on Elasticsearch. Since results are expected to be larged, batched pickle files are generated

    Args:
        snapshot_path (str, optional): Path of directory that stores snapshots. Defaults to ''.
    Returns:
        df (pd.DataFrame): A concatenated dataframe containing all the specified columns.
    """

    snapshot_path = Path(snapshot_path)
    snapshot_path.mkdir(exist_ok=True,parents=True)

    es = Elasticsearch(
        hosts=[elastic_config['hostname']],
        verify_certs=False,
    )    
    # we will add a query to only grab the ones that contain at least one keyword (or partially, if keyword was space separated)
    mainquery = elastic_config['mainquery']
    mainquery['query']['bool']['must'] = {
        "simple_query_string": {
            "query": ' '.join(keywords),
            "fields": [
                "text",
                "extended_tweet.full_text"
            ],
        }
    }
    doc_count = es.count(
        index=[elastic_config['index_name']],
        body=mainquery,
        request_timeout = 120,
    )['count']


    # snapshots = sorted()
    df_list = []

    # if there are no snapshots or the last (sorted) snapshot does not match the doc count, start over
    if not any(snapshot_path.glob('*.pkl')):
        cursor = scan(
            es,
            index=elastic_config['index_name'],
            query = {**mainquery, '_source':elastic_config['fields']},
            size=10000,
            request_timeout = 120,
        )
        results = [c for c in tqdm(cursor,total=doc_count,desc='scrolling index')]
        print('turning into dataframe...')
        df = prettify_elastic(results)
        for c in tqdm(df.columns,desc='saving dataframe...'):
            df[c].to_pickle(snapshot_path/f'{c}.pkl')

        
    # we have everything, just need to concat the dataframes
    else:
        print('loading from cache...')
        df = pd.DataFrame([
            pd.read_pickle(f)
            for f in snapshot_path.glob('*.pkl')
        ])

    return df


def get_snapshot_column(snapshot_path,column_name):
    column_path = Path(snapshot_path) / f'{column_name}.pkl'
    if not column_path.is_file():
        df = cache_index(snapshot_path)
        return df[column_name]
    else:
        print('column already created. loading...')
        return pd.read_pickle(column_path)