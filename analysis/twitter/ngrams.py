from tqdm.auto import tqdm
from collections import Counter
from itertools import chain
from tokenizer import load_tokens
from wordcloud import WordCloud
from skimage.draw import disk
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm


def make_circle(radius=1000):
    """Hepler function to draw circle in ngram wordcloud background. Note that the wordcloud background can be changed to any other shape

    Args:
        radius (int, optional): radius of circle. In pixels. Defaults to 1000.

    Returns:
        _type_: A square canvas with a circle filled in black
    """
    canvas = np.ones((radius*2+1, radius*2+1), dtype=np.uint8)*255
    canvas[disk((radius, radius), radius)] = 0
    return canvas


cloud_shape = make_circle(1000)


def count_grams(tokens=[], n=1,separater='',ngram_path = ''):
    """Counts frequency of ngrams. Returns a list of tuples in descending order

    Args:
        tokens (list, optional): List of List of strings. Each list of strings represents a text. Defaults to [].
        n (int, optional): n in ngrams to search for. Defaults to 1.
        separater (int, optional): separater for the words in a gram. Useful for drawing
        ngram_path (string, optional): to save the grams

    Returns:
        _type_: _description_
    """

    

    gram_counts = Counter(
        chain(*map(
            lambda toks: map(
                lambda gram: separater.join(gram), 
                zip(*[toks[i:] for i in range(n)])
            ),
            tqdm(tokens,desc='making grams..',total=len(tokens),leave=False)
        ))
    )

    by_count = sorted(gram_counts.items(), key=lambda x: x[1], reverse=True)

    dict(by_count)

    return by_count


def draw_ngrams(
    n=3,
    num=500,
    token_path='',
    plots_path='plots',
    mode='mixed',
    separater = '',
):
    """Draws ngram wordclouds

    Args:
        n (int, optional): maximum number of n to go up. Defaults to 3.
        num (int, optional): maximum number to include in wordcloud. Defaults to 500.
        tokens_path (str, optional): path to load tokens from. Defaults to ''.
        plots_path (str, optional): folder to save output images in. Defaults to 'plots'.
        mode (str,optional): ['mixed','emoji','text']. Defaults to 'mixed'.
    """
    # tokens = load_tokens(tokens_path)
    pfold = Path(plots_path)
    pfold.mkdir(parents=True,exist_ok=True)

    # if mode == 'emoji':
    #     tokens = list(get_emojis(tokens))
    tokens = load_tokens(Path(token_path)/mode)

    for i in range(1, n+1):
        fig, ax = plt.subplots(figsize=(8, 8))
        by_count = count_grams(tokens, i, separater)
        wc = WordCloud(background_color="white", max_words=num,
                       mask=cloud_shape, max_font_size=400, font_path='./Symbola.otf')
        wc.generate_from_frequencies(dict(by_count[:num]))
        ax.imshow(wc)
        ax.axis('off')
        fig.savefig(pfold/f'{mode}_{i}_grams.png', dpi=300)
        print(f'{mode} -- {i}grams drawn')

import emoji
