from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
import json
from tqdm.auto import tqdm
from pathlib import Path


def scroll_index(
    hostname = '',
    index_name = '',
    fields = []
):
    """scroll_index

    Args:
        hostname (str, optional): Address to access ElasticSearch host at. Defaults to ''.
        index_name (str, optional): Name of index to send queries to. Defaults to ''.
        fields (list, optional): Fields of index to grab. Defaults to [].

    Returns:
        cursor (generator): generator object that returns a single Hit on iteration.
        doc_count (int): number of documents found in the index. Used for keeping track of progress.
    """
    es = Elasticsearch(
        hosts=[hostname],
        verify_certs=False,
        # timeout=config.elasticsearch_timeout_secs
    )    
    mainquery = {
        'query': {
            # 'match_all': {},
            'bool': {
                'must_not': {
                    'exists': {
                        'field': 'retweeted_status.id'
                    }
                }
            },
        },
    }

    doc_count = es.count(
        index=['blockchain-interoperability-attacks'],
        body=mainquery,
    )['count']

    cursor = scan(
        es,
        index=index_name,
        query = {**mainquery, '_source':fields},
        size=10000
    )

    return cursor,doc_count

def prettify_elastic(results):
    """prettify_elastic

    Args:
        results (_type_): List of dataframes with a shared column.

    Returns:
        df (pd.DataFrame): Concatenated dataframe with indexes reset
    """
    df = pd.json_normalize(results)
    df = df.drop(columns=['_index','_type','_id','_score','sort','_ignored'])
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

def get_fields_from_index(
    hostname = '',
    index_name = '',
    snapshot_name = '',
    batch_size=100000,
    fields = [],
):
    """get_fields_from_index

    Args:
        hostname (str, optional): Address to access ElasticSearch host at. Defaults to ''.
        index_name (str, optional): Name of index to send queries to. Defaults to ''.
        snapshot_name (str, optional): Path of directory that stores snapshots. Defaults to ''.
        batch_size (int, optional): Maximum length of each json file. Defaults to 100000.
        fields (list, optional): Fields of index to grab. Defaults to [].

    Returns:
        df (pd.DataFrame): A concatenated dataframe containing all the specified columns.
    """
    snapshots_folder = Path(snapshot_name)
    snapshots_folder.mkdir(exist_ok=True,parents=True)

    cursor,doc_count = scroll_index(hostname,index_name,fields)

    df_list = []
    if (not any(snapshots_folder.iterdir())) or (sorted(snapshots_folder.glob('*.json'))[-1].stem != f'{doc_count-1:08d}'):
        results = []
        for i,c in enumerate(tqdm(cursor,total=doc_count)):
            if i > 0 and i % batch_size == 0:
                df = prettify_elastic(results)
                df.to_json(snapshots_folder/f'{i:08d}.json','records')
                df_list.append(df)
                del results
                results = [] 

            results.append(c)

        df = prettify_elastic(results)
        df.to_json(snapshots_folder/f'{i:08d}.json','records')
        df_list.append(df)
        del results
        df = pd.concat(df_list).reset_index(drop=True)
    else:
        # df = pd.DataFrame()
        df = load_cache(snapshots_folder)
        # for files in sorted(snapshots_folder.glob('*.csv')):
            # batch = pd.read_csv(snapshot_file)

    return df

def load_cache(snapshots_path):
    """load_cache

    Args:
        snapshots_path (str): Path of directory that stores snapshots

    Returns:
        df (pd.DataFrame): A concatenated dataframe containing all the specified columns.
    """
    snapshots_folder = Path(snapshots_path)
    df_list = [pd.read_json(f) for f in tqdm(sorted(snapshots_folder.glob('*.json')))]
    return pd.concat(df_list).reset_index(drop=True)


if __name__ == '__main__':
    get_fields_from_index(
        'http://idea-vm-elasticsearch:9200',
        'blockchain-interoperability-attacks',
        '/data/blockchain-interoperability/blockchain-social-media/analysis/twitter/snapshots',
        int(1e5),
        [
            'id',
            'created_at',
            'timestamp_ms',
            'user.id',
            'user.description',
            'text',
            'entities.hashtags',
            # 'entities.urls',
            # 'entities.user_mentions',
            'truncated',
            'extended_tweet.full_text',
            'extended_tweet.entities.hashtags',
            # 'extended_tweet.entities.urls',
            # 'extended_tweet.entities.user_mentions',
            'quote_count',
            'reply_count',
            'retweet_count',
            'favorite_count',
            'lang',
            'in_reply_to_status_id',
            'n_reply_to_user_id',
        ]
    )