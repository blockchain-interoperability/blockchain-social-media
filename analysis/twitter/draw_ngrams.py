from tqdm.auto import tqdm
from collections import Counter
from itertools import chain
from tokenizer import load_tokens
from wordcloud import WordCloud
from skimage.draw import disk
import matplotlib.pyplot as plt
import numpy as np


tokens = load_tokens('/data/blockchain-interoperability/blockchain-social-media/analysis/twitter/tokens')


from tqdm.auto import tqdm

def extract_grams(tokens, n):
    # inp = sentence.split()
    return Counter([' '.join(gram)
                    for gram in chain(*[zip(*[toks[i:] for i in range(n)]) for toks in tqdm(tokens, desc=f'generating {n} grams', leave=False)])])

# gram_counts = Counter()
# for toks in tqdm(tokens):
gram_counts_1 = extract_grams(tokens,1)
gram_counts_2 = extract_grams(tokens,2)
gram_counts_3 = extract_grams(tokens,3)


by_count_1 = sorted(gram_counts_1.items(),key=lambda x: x[1],reverse=True)
by_count_2 = sorted(gram_counts_2.items(),key=lambda x: x[1],reverse=True)
by_count_3 = sorted(gram_counts_3.items(),key=lambda x: x[1],reverse=True)

def make_circle(radius):
    canvas = np.ones((radius*2+1, radius*2+1), dtype=np.uint8)*255
    canvas[disk((radius, radius), radius)] = 0
    return canvas

# the regex used to detect words is a combination of normal words, ascii art, and emojis
# 2+ consecutive letters (also include apostrophes), e.x It's
# normal_word = r"(?:\w[\w']+)"
# # 2+ consecutive punctuations, e.x. :)
# ascii_art = r"(?:[{punctuation}][{punctuation}]+)".format(punctuation=string.punctuation)
# # a single character that is not alpha_numeric or other ascii printable
# emoji = r"(?:[^\s])(?<![\w{ascii_printable}])".format(ascii_printable=string.printable)
# regexp = r"{normal_word}|{ascii_art}|{emoji}".format(normal_word=normal_word, ascii_art=ascii_art,
#                                                      emoji=emoji)
# d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
# fpath = path.join(d, 'Apple Color Emoji.ttc')


cloud_shape = make_circle(1000)
wc = WordCloud(background_color="white", max_words=500, mask=cloud_shape,max_font_size=400,font_path='./Symbola.otf')
wc.generate_from_frequencies(dict(by_count_1[:500]))

fig,ax = plt.subplots(figsize=(8,8))
ax.imshow(wc)
ax.axis('off')
fig.savefig('1_gram_test.png',dpi=300)
print('1 grams drawn')

fig,ax = plt.subplots(figsize=(8,8))
wc = WordCloud(background_color="white", max_words=500, mask=cloud_shape,max_font_size=400,font_path='./Symbola.otf')
wc.generate_from_frequencies(dict(by_count_2[:500]))
ax.imshow(wc)
ax.axis('off')
fig.savefig('2_gram_test.png',dpi=300)
print('2 grams drawn')

fig,ax = plt.subplots(figsize=(8,8))
wc = WordCloud(background_color="white", max_words=500, mask=cloud_shape,max_font_size=400,font_path='./Symbola.otf')
wc.generate_from_frequencies(dict(by_count_3[:500]))
ax.imshow(wc)
ax.axis('off')
fig.savefig('3_gram_test.png',dpi=300)
print('3 grams drawn')