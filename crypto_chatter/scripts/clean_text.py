from crypto_chatter.data.text import clean_text, extract_hashtags
from crypto_chatter.utils import progress_bar
from crypto_chatter.data import CryptoChatterData
from crypto_chatter.config import (
    CryptoChatterDataConfig,
)

import pandas as pd

dataset = 'twitter:blockchain-interoperability-attacks'
graph_type = 'tweet'
data_config = CryptoChatterDataConfig(dataset)
snapshots = sorted(data_config.raw_snapshot_dir.glob('*.pkl'))
with progress_bar() as progress:
    snotshot_task = progress.add_task('Cleaning snapshots..', total=len(snapshots))
    for snapshot in snapshots:
        df = pd.read_pickle(snapshot)
        # print('loaded snapshot', snapshot.name)


        has_bad_col = 'truncatedquoted_status.truncated' in df.columns
        needs_stacking = any(df.columns.str.contains('quoted_status'))
        already_cleaned = 'clean_text' in df.columns
        already_tagged = 'hashtags' in df.columns

        # print(df.columns)
        # print(has_bad_col, needs_stacking, already_cleaned, already_tagged)
        # quit()

        
        if has_bad_col:
            del df['truncatedquoted_status.truncated']

        if needs_stacking:
            quoted_cols = df.columns[df.columns.str.contains('quoted_status.')]
            regular_cols = df.columns[~df.columns.str.contains('quoted_status.')]
            quote_has_text = (~df['quoted_status.text'].isna() & ~df['quoted_status.extended_tweet.full_text'].isna())
            quoted_df = df[quote_has_text][quoted_cols]
            quoted_df.columns = quoted_df.columns.str.replace('quoted_status.', '')
            df = pd.concat([df[regular_cols], quoted_df])

        cleaned_text = []
        hashtags = []
        if (not already_cleaned) or (not already_tagged):
            clean_task = progress.add_task('Cleaning text..', total=len(df))
            for text in df[data_config.text_col].values:
                if not already_cleaned:
                    cleaned_text += [clean_text(text)]
                if not already_tagged:
                    hashtags += [extract_hashtags(text)]
                progress.advance(clean_task)
            progress.remove_task(clean_task)

        if not already_cleaned:
            df['clean_text'] = cleaned_text
        if not already_tagged:
            df['hashtags'] = hashtags

        if any([has_bad_col, needs_stacking, already_cleaned, already_tagged]):
            df.to_pickle(snapshot)
        progress.advance(snotshot_task)
        del df

    data = CryptoChatterData(
        data_config = data_config,
        columns = ['hashtags'],
        progress=progress,
    )
