import pandas as pd
from rich.progress import Progress

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import extract_hashtags

from .text import extract_hashtags, clean_text

def prettify_elastic_twitter(
    results:list[dict], 
    data_config:CryptoChatterDataConfig,
    progress: Progress|None = None,
) -> pd.DataFrame:
    """Cleans up list of results into a dataframe while dropping some unncessary columns.

    Args:
        results (_type_): List of dataframes with a shared column.

    Returns:
        df (pd.DataFrame): Concatenated dataframe with indexes reset
    """
    if progress is not None:
        parse_task = progress.add_task("Parsing df..", total=4)

    df = pd.json_normalize(results)
    df = df.drop(columns=df.columns[~df.columns.str.contains('_source')])
    df.columns = df.columns.str.replace('_source.','')
    if progress is not None:
        progress.advance(parse_task)

    # If trucated is False, that means text has the full text. If True, extended_tweet.full_text has the full text.
    df['truncated'] = df['truncated'].fillna(False)
    df[data_config.text_col] = df['text']
    df.loc[df['truncated'],data_config.text_col] = df[df['truncated']]['extended_tweet.full_text']
    if progress is not None:
        progress.advance(parse_task)

    df['quoted_status.truncated'] = df['quoted_status.truncated'].fillna(False)
    df['quoted_status.full_text'] = df['quoted_status.text']
    df.loc[df['quoted_status.truncated'],'quoted_status.full_text'] = df[df['quoted_status.truncated']]['quoted_status.extended_tweet.full_text']
    if progress is not None:
        progress.advance(parse_task)

    # find quoted tweets that have valid text and add as well.
    quoted_cols = df.columns[df.columns.str.contains('quoted_status.')]
    regular_cols = df.columns[~df.columns.str.contains('quoted_status.')].tolist() + ['quoted_status.id']
    quote_has_text = (~df['quoted_status.text'].isna() & ~df['quoted_status.extended_tweet.full_text'].isna())
    quoted_df = df[quote_has_text][quoted_cols]
    quoted_df.columns = quoted_df.columns.str.replace('quoted_status.', '')
    df = pd.concat([df[regular_cols], quoted_df])
    if progress is not None:
        progress.advance(parse_task)

    # parse hashtags and clean text
    hashtags = []
    cleaned_text = []
    if progress is not None:
        clean_task = progress.add_task("cleaning text..", total=len(df))
    for text in df[data_config.text_col].values:
        hashtags += [extract_hashtags(text)]
        cleaned_text += [clean_text(text)]

        if progress is not None:
            progress.advance(clean_task)

    df['hashtags'] = hashtags
    df['clean_text'] = cleaned_text

    if progress is not None:
        progress.remove_task(parse_task)
        progress.remove_task(clean_task)

    return df
