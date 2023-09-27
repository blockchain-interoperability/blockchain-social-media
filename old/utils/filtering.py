import pandas as pd
from pathlib import Path
import json
from tqdm.auto import tqdm

# very simple keyword checker. This alone gets reduces size by ~50%
keywords = json.load(open('utils/keywords.json'))
def has_keyword(text):
    return any(
        text.count(k) > 0
        for k in keywords
    )


def filter_data(
    snapshot_path = ''
):
    # load the entire snapshot, apply filter, save the filtered ones in diff directory
    snapshot_path = Path(snapshot_path)
    # huge!
    whole_data = pd.concat([
        pd.read_pickle(f)
        for f in tqdm(sorted(snapshot_path.glob('*.pkl')),desc='loading entire dataset..')
    ])
    good_rows = whole_data[whole_data['whole_text'].apply(has_keyword)]

    

