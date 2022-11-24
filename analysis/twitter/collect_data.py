from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path


def scroll_index(
    es,
    index_name = '',
    mainquery = {'query':{'match_all':{}}},
    fields = [],
):
    """Sets up up scrolling request for the index

    Args:
        es (ElasticSearch): Instance of ElasticSearch client
        index_name (str, optional): Name of index to send queries to. Defaults to ''.
        mainquery (dict,optional): Main query to use with Elasticsearch. Defaults to {'query':{'match_all':{}}}.
        fields (list, optional): Fields of index to grab. Defaults to [].

    Returns:
        cursor (generator): generator object that returns a single Hit on iteration.
    """
    cursor = scan(
        es,
        index=index_name,
        query = {**mainquery, '_source':fields},
        size=10000
    )

    return cursor

def prettify_elastic(results):
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
    df['whole_text'] = df[
        [
            'extended_tweet.full_text',
            'text',     
            'truncated',
        ]
    ].apply(
        lambda row: row[0] if row[2] == True else row[1] if row[2] == False else '',
        axis=1
    )
    return df    

def cache_index(
    hostname = '',
    index_name = '',
    snapshot_path = '',
    batch_size=100000,
    mainquery = {'query':{'match_all':{}}},
    sort = {'':''},
    fields = [],
):
    """Grabs the specified fields from the specified index on Elasticsearch. Since results are expected to be larged, batched pickle files are generated

    Args:
        hostname (str, optional): Address to access ElasticSearch host at. Defaults to ''.
        index_name (str, optional): Name of index to send queries to. Defaults to ''.
        snapshot_path (str, optional): Path of directory that stores snapshots. Defaults to ''.
        batch_size (int, optional): Maximum length of each pickle file. Defaults to 100000.
        mainquery (dict, optional): Main query to use with Elasticsearch. Defaults to {'query':{'match_all':{}}}.
        sort (dict, optional): Operation to sort the query results by. Excluded from main query because count doesn't support it.
        fields (list, optional): Fields of index to grab. Defaults to [].

    Returns:
        df (pd.DataFrame): A concatenated dataframe containing all the specified columns.
    """
    es = Elasticsearch(
        hosts=[hostname],
        verify_certs=False,
        # timeout=config.elasticsearch_timeout_secs
    )    

    doc_count = es.count(
        index=[index_name],
        body=mainquery,
    )['count']

    snapshot_folder = Path(snapshot_path)
    snapshot_folder.mkdir(exist_ok=True,parents=True)
    snapshots = sorted(snapshot_folder.glob('*.pkl'))
    df_list = []
    if (not any(snapshots)) or (snapshots[-1].stem != f'{doc_count-1:08d}'):
        
        # offset = 0
        # if len(snapshots) >= 2:
        #     last_result = pd.read_pickle(snapshots[-2])
        #     offset = int(snapshots[-2].stem)
        #     last_timestamp = pd.to_datetime(last_result['timestamp_ms'],unit='ms')[-1].strftime('%Y-%m-%dT%H:%M:%S.%f')
        #     mainquery['query']['bool']['filter']['range']['created_at']['gte'] = last_timestamp
        #     snapshots[-1].unlink()
        # mainquery['sort'] = sort
        cursor = scroll_index(es,index_name,mainquery,fields)


        results = []
        for i,c in enumerate(tqdm(cursor,total=doc_count)):
            if i > 0 and i % batch_size == 0 or i == doc_count-1:
                df = prettify_elastic(results)
                df.to_pickle(snapshot_folder/f'{i:08d}.pkl')
                df_list.append(df)
                del results
                results = [] 

            results.append(c)
        df = pd.concat(df_list).reset_index(drop=True)
    else:
        df = load_cache(snapshot_folder)
    
    return df
    

def load_cache(snapshot_path = ''):
    """Aggregates the cached pickle files and returns a single DataFrame

    Args:
        snapshot_path (str): Path of directory that stores snapshots

    Returns:
        df (pd.DataFrame): A concatenated dataframe containing all the specified columns.
    """
    snapshot_folder = Path(snapshot_path)
    df_list = [pd.read_pickle(f) for f in tqdm(sorted(snapshot_folder.glob(f'*.pkl')),leave=False,desc='reading in snapshots..')]
    return pd.concat(df_list).reset_index(drop=True)


