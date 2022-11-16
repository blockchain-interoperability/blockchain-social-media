import pandas as pd
from pathlib import Path
import pickle
from tqdm.auto import tqdm
import emoji



def get_emoji_tokens(tokens=[]):
    """Gets only emoji characters from the tokens

    Args:
        tokens (list, optional): cleaned tokens. Each emoji is expected to be in a separate string. Defaults to [].

    Returns:
        list[list[str]]: list of list of strings. each string is an emoji.
    """
    # emojis_list = []
    return map(
        lambda toks: list(set(
            filter(
                lambda t: t in emoji.UNICODE_EMOJI.keys(), 
                toks
            )
        )),
        tokens
    )

def get_text_tokens(tokens=[]):
    """Gets only non-emoji words from the tokens

    Args:
        tokens (list, optional): cleaned tokens. Each word is expected to be in a separate string. Defaults to [].

    Returns:
        list[list[str]]: list of list of strings. each string is an word].
    """
    return map(
        lambda toks: list(set(
            filter(
                lambda t: not t in emoji.UNICODE_EMOJI.keys(), 
                toks
            )
        )),
        tokens
    )

custom_filter = lambda tok: not (tok.is_stop or tok.is_punct or tok.is_space or tok.like_url  or '@' in str(tok))

def get_tokens(nlp,text_list): 
    """Turns a list of text into list of tokens. Separated out from the main function for debugging and scaling purpose

    Args:
        nlp (any): Spacy nlp instance.
        text_list (List[str],pd.Series[str]): Iterable containing text.

    Returns:
        _type_: List[List[str]]: List of tokens for each text
    """

    parsed = nlp.pipe(text_list, disable=["tok2vec", "parser","ner"])
    filtered_lemmas = list(map(
        lambda doc: list(map(lambda tok: tok.lemma_, filter(custom_filter,doc))),
        parsed
    ))

    del parsed
    return {
        'mixed':filtered_lemmas, 
        'emoji':get_emoji_tokens(filtered_lemmas), 
        'text':get_text_tokens(filtered_lemmas)
    }
    
def tokenize_text(
    snapshot_path = '',
    # cache_path={'all': '','emoji': '', 'text':''},
    token_path= '',
    minibatch=1000,
    token_types = ['mixed','emoji','text'],
):
    """Creates list of tokens per text and saves it in batched pickle files.

    Args:
        snapshot_path (str, optional): Path to read snapshots from. Defaults to ''.
        token_path (str, optional): Path to save the token lists to. Defaults to ''.
        minibatch (int, optional): Size of mini batches to use with Spacy. Defaults to 1000
    """
    import spacy
    from spacymoji import Emoji

    spacy.require_gpu()
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('emoji', first=True)

    snapshot_folder = Path(snapshot_path)

    tokens_cache = dict(zip(token_types,map(lambda t: Path(token_path)/t,token_types)))
    for c in tokens_cache.values(): c.mkdir(parents=True,exist_ok=True)

    # check if files already exist. then we don't need to re-run
    # (not any([any(t.iterdir()) for t in tokens_cache.values()])) or (sorted(snapshot_folder.glob('*.pkl'))[-1].stem != f'{doc_count-1:08d}')    

    docs_count =sorted(snapshot_folder.glob('*.pkl'))[-1].stem
    is_up_to_date = all([
        any(t.glob('*.pkl')) and sorted(t.glob('*.pkl'))[-1].stem == docs_count
        for t in tokens_cache.values()
    ])
    if not is_up_to_date:
        for f in tqdm(sorted(snapshot_folder.glob(f'*.pkl')),desc='Generating tokens...',leave=False): 
            # token_file = token_folder/f'{f.stem}.pkl'
            # if token_file.is_file(): pass

            df = pd.read_pickle(f)
            
            tokens = dict(map(lambda t: (t,[]), token_types))
            for i in tqdm(range(0,len(df),minibatch),desc=f'batch {f.stem}', leave=False):
                
                all_tokens = get_tokens(nlp,df.whole_text.values[i:i+minibatch])
                for k,v in tokens.items():
                    v += all_tokens[k]

            for k,v in tokens.items():
                pickle.dump(v,open(tokens_cache[k]/f'{f.stem}.pkl','wb'))
            del tokens,df


def load_tokens(
    token_path = '',
    show_progress = True,
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
    for f in tqdm(sorted(tokens_folder.glob('*.pkl')),desc='loading tokens...',leave = False,disable = not show_progress):
        tokens += pickle.load(open(f,'rb'))
    return tokens


# def extract_emojis(

# )