from sentence_transformers import SentenceTransformer
from pathlib import Path
from tokenizer import load_tokens
import numpy as np
# import umap
import torch
import pickle
from tqdm.auto import tqdm

def get_sbert_embedding(tokens, embedder):
    model = SentenceTransformer(embedder,device='cuda')
    
    def encode(minibatch):
        e = model.encode(list(map(lambda x: ' '.join(x),minibatch)), show_progress_bar=False,convert_to_tensor=True)
        e_cpu = e.cpu()
        del e
        return e_cpu

    batch_embeddings = torch.vstack([
        encode(tokens[i:i+1000] )
        for i in tqdm(
            range(0,len(tokens),1000),
            leave=False,
            desc='minibatch..'
        )
    ] )

    batch_embeddings_cpu = batch_embeddings.cpu()
    del batch_embeddings,model
    return batch_embeddings_cpu


def create_sbert_embeddings(embedder = 'all-MiniLM-L6-v2',tokens_path = '',bert_embeddings_path = ''):
    """Creates bert embeddings. Each list of tokens is joined to a string and fed into the bert transformer to create an embedding

    Args:
        tokens_path (str, optional): Path of cached tokens. Defaults to ''.
        bert_embeddings_path (str, optional): Path to save the results in. Defaults to ''.

    Returns:
        list[list[int]]: embeddings generated from bert
    """
    embeddings_folder = Path(bert_embeddings_path) / embedder
    embeddings_folder.mkdir(exist_ok=True,parents=True)
    created_embeddings = sorted(embeddings_folder.glob('*.pkl'))
    token_batches = sorted(Path(tokens_path).glob('*.pkl'))
    
    if len(created_embeddings) == len(token_batches):
        embeddings = torch.vstack([torch.load(s) for s in created_embeddings])
        print('loaded cached bert embeddings!')
    else:
        start_index = 0
        embeddings = []
        if len(created_embeddings) > 1:
            start_index = len(created_embeddings)-1
            embeddings = [torch.load(s) for s in created_embeddings]
            print(f'picking up from {start_index}')
        # tokens = load_tokens(tokens_path)
        for batch in tqdm(
            # list(filter(lambda f: int(f.stem) >= int(start_index),token_batches)),
            # range(start_index,len(token_batches), 1000),
            token_batches[start_index:],
            desc='creating embeddings',
            leave=False
        ):  
            tokens = pickle.load(open(batch,'rb'))
            batch_embeddings = get_sbert_embedding(tokens,embedder)

            torch.save(batch_embeddings,embeddings_folder / batch.name)
            embeddings.append(batch_embeddings)
            
            del tokens
            torch.cuda.empty_cache()
            
        embeddings = torch.vstack(embeddings)
    return embeddings

# dimensionality reduction
# def get_umap_embeddings(n_neighbors = 15,n_components = 5,tokens_path = '',bert_embeddings_path='',umap_embeddings_path = ''):
#     """Returns UMAP reduced embeddings of BERT embeddings.

#     Args:
#         tokens_path (str, optional): Path of cached tokens. Defaults to ''.
#         bert_embeddings_path (str, optional): Path to save the bert embeddings in. Defaults to ''.
#         umap_embeddings_path (str, optional): Path to save the UMAP embeddings in. Defaults to ''.

#     Returns:
#         list[list[int]]: bert embeddings reduced through bert
#     """
#     embeddings = get_bert_embeddings(tokens_path,bert_embeddings_path)
#     umap_embeddings_file = Path(umap_embeddings_path)
#     umap_embeddings_file.parent.mkdir(exist_ok=True,parents=True)
#     if not umap_embeddings_file.is_file():
#         umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, 
#                                     n_components=n_components, 
#                                     metric='cosine').fit_transform(embeddings)
#         np.save(umap_embeddings_file,umap_embeddings)
#     else:
#         umap_embeddings = np.load(umap_embeddings_file)
#     return umap_embeddings


def load_embeddings(
    embedding_path
):
    return torch.vstack([
        torch.load(f) 
        for f in tqdm(sorted(Path(embedding_path).glob('*.pkl')),desc='loading embeddings..',leave=False)
    ])