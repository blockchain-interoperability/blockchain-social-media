# Twitter data analysis

All the heavy operations should be ran through main.py. Only once the cache is created locally should you use the data on jupyter.

## Analysis

- NGram analysis
- Topic clustering
    - Tough to reduce dimension on such large data, will use autoencoder to reduce dimension and use KNN or something to find clusters. Then for each cluster, we can find common words that appear the most, which should describe the topics of each cluster
- What else? 



## main.py commands
```
python3 main.py [funcname]
```
The arguments for the function can be saved in (twitter_config.json)[twitter_config.json]

If you want the use the default values from the config.json file, you can also use pass the entire argument to every function
```python
import json
config = json.load(open('config.json'))
df = cache_index(**config)
```

### Loading the newst data

The entry point to Elasticsearch is by grabbing the interested fields and downloading a dataframe. These files are stored under `snapshot/`
This command also automatically finds valid text for any tweet and saves it under `whole_text` column.
```
python3 main.py cache_index
```

To load in jupyter:
```python
from utils.collect_data import cache_index
df = cache_index('/data/blockchain-interoperability/blockchain-social-media/analysis/twitter/snapshots')
```

You can also access the individual columns of the dataframe (quicker) by loading from the cache folder, saved under `{column_name}.pkl`

```python
from utils.collect_data import cache_index
whole_text = cache_index('/data/blockchain-interoperability/blockchain-social-media/analysis/twitter/snapshots/whole_text.pkl')
```

### Loading embeddings



### Tokenizing the text

The tokenizer should be ran after the latest cache is created. 
Spacy's tokenizer and lemmatizer pipeline is used for this process. Once the text is stripped, it saves the tokens through 3 different filters. `mixed`, `emoji`, and `text`.

```
python3 main.py tokenize_text
```

To load filtered tokens, you can

```python
from utils.tokenizer import get_tokens
mode = 'text'
tokens = get_tokens(f'/data/blockchain-interoperability/blockchain-social-media/analysis/twitter/tokens/{mode}')
```



### Drawing Ngrams

You can count and plot ngrams once the tokenization is done. 
For example, the text only one can be drawn like this:
```
python3 main.py draw_ngrams_text
```
Note that in the settings, `n = 3`. This means that 1 and 2 grams will also be counted and plotted.


If you want the ngrams by count, you can run
```python
from utils.tokenizer import get_tokens
from utils.ngrams import count_grams

mode = 'text'
tokens = get_tokens(f'/data/blockchain-interoperability/blockchain-social-media/analysis/twitter/tokens/{mode}')
count_grams(tokens,separater = ' ')
```

Note that the grams are joined by a separater when returned