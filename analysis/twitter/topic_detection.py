from sentence_transformers import SentenceTransformer
from pathlib import Path
from tokenizer import load_tokens
import numpy as np
import umap
import torch

def get_bert_embeddings(tokens_path = '',bert_embeddings_path = '',):
    """Creates bert embeddings. Each list of tokens is joined to a string and fed into the bert transformer to create an embedding

    Args:
        tokens_path (str, optional): Path of cached tokens. Defaults to ''.
        bert_embeddings_path (str, optional): Path to save the results in. Defaults to ''.

    Returns:
        list[list[int]]: embeddings generated from bert
    """
    bert_embeddings_file = Path(bert_embeddings_path)
    bert_embeddings_file.parent.mkdir(exist_ok=True,parents=True)
    if not bert_embeddings_file.is_file():
        tokens = load_tokens(tokens_path)
        model = SentenceTransformer('all-MiniLM-L6-v2',device='cuda')
        embeddings = model.encode(list(map(lambda x: ' '.join(x),tokens)), show_progress_bar=True,convert_to_tensor=True)
        torch.save(embeddings,bert_embeddings_file)
    else:
        embeddings = torch.load(bert_embeddings_file)
        print('loaded cached bert embeddings!')
    return embeddings

# dimensionality reduction
def get_umap_embeddings(n_neighbors = 15,n_components = 5,tokens_path = '',bert_embeddings_path='',umap_embeddings_path = ''):
    """Returns UMAP reduced embeddings of BERT embeddings.

    Args:
        tokens_path (str, optional): Path of cached tokens. Defaults to ''.
        bert_embeddings_path (str, optional): Path to save the bert embeddings in. Defaults to ''.
        umap_embeddings_path (str, optional): Path to save the UMAP embeddings in. Defaults to ''.

    Returns:
        list[list[int]]: bert embeddings reduced through bert
    """
    embeddings = get_bert_embeddings(tokens_path,bert_embeddings_path)
    umap_embeddings_file = Path(umap_embeddings_path)
    umap_embeddings_file.parent.mkdir(exist_ok=True,parents=True)
    if not umap_embeddings_file.is_file():
        umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, 
                                    n_components=n_components, 
                                    metric='cosine').fit_transform(embeddings)
        np.save(umap_embeddings_file,umap_embeddings)
    else:
        umap_embeddings = np.load(umap_embeddings_file)
    return umap_embeddings


