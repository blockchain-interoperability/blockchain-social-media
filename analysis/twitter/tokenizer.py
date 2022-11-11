import pandas as pd
from pathlib import Path
import spacy
# import json
import pickle
from tqdm.auto import tqdm

# spacy.require_gpu()
nlp = spacy.load('en_core_web_lg')

custom_filter = lambda tok: not (tok.is_stop or tok.is_punct or tok.is_space or tok.like_url  or '@' in str(tok))

def get_tokens(text_list): 
    """Turns a list of text into list of tokens. Separated out from the main function for debugging and scaling purpose

    Args:
        text_list (List[str],pd.Series[str]): Iterable containing text

    Returns:
        _type_: List[List[str]]: List of tokens for each text
    """
    return list(map(
        lambda doc: list(map(lambda tok: tok.lemma_, filter(custom_filter,doc))),
        nlp.pipe(text_list, disable=["tok2vec", "parser","ner"])
    ))

def tokenize_text(
    snapshot_path = '',
    token_path='',
    minibatch=1000,
):
    """Creates list of tokens per text and saves it in batched pickle files.

    Args:
        snapshot_path (str, optional): Path to read snapshots from. Defaults to ''.
        token_path (str, optional): Path to save the token lists to. Defaults to ''.
        minibatch (int, optional): Size of mini batches to use with Spacy. Defaults to 1000
    """
    snapshot_folder = Path(snapshot_path)
    token_folder = Path(token_path)

    token_folder.mkdir(parents=True,exist_ok=True)
    for f in tqdm(sorted(snapshot_folder.glob(f'*.pkl')),desc='Generating tokens...',leave=False): 
        df = pd.read_pickle(f)
        
        tokens = []
        for i in tqdm(range(0,len(df),minibatch),desc=f'batch {f.stem}', leave=False):
            tokens += get_tokens(df.whole_text[i:i+minibatch])

        pickle.dump(tokens,open(token_folder/f'{f.stem}.pkl','wb'))
        del tokens,df


def load_tokens(
    token_path = '',
):
    """Loads tokens from cache

    Args:
        tokens_path (str, optional): Path of saved token files. Defaults to ''.

    Returns:
        tokens (List[List[str]]): list of list of tokens. List of tokens per text.
    """
    tokens_folder = Path(token_path)
    # check if lemmas match the cache
    tokens = []
    for f in sorted(tokens_folder.glob('*.pkl')):
        tokens += pickle.load(open(f,'rb'))
    return tokens