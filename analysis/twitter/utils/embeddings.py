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
        if created_embeddings:
            start_index = len(created_embeddings)-1
            embeddings = [torch.load(s) for s in created_embeddings]
            print(f'picking up from {start_index}')
        for batch in tqdm(
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

def load_embeddings(
    embedding_path
):
    return torch.vstack([
        torch.load(f) 
        for f in tqdm(sorted(Path(embedding_path).glob('*.pkl')),desc='loading embeddings..',leave=False)
    ])