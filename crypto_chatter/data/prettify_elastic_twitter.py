import pandas as pd

from crypto_chatter.config import CryptoChatterDataConfig

def prettify_elastic_twitter(results:list[dict], data_config:CryptoChatterDataConfig) -> pd.DataFrame:
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
            ] + data_config.es_columns
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
