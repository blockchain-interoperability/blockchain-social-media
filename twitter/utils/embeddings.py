from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
# import umap
import torch
import pickle
from tqdm.auto import tqdm

from utils.collect_data import get_snapshot_column
# from utils.tokenizer import load_tokens

def batch_transformer_embeddings(tokens, embedder):
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


def get_transformer_embeddings(
    embedder = 'all-MiniLM-L6-v2',
    snapshot_path = '',
    embeddings_path = '',
    **kwargs
    # embedder_batch_size = 100000
):
    """Creates bert embeddings. Each list of tokens is joined to a string and fed into the bert transformer to create an embedding

    Args:
        embedder (str, optional): embedder name. Defaults to 'all-MiniLM-L6-v2'
        snapshot_path (str, optional): Path of cached snapshot path. Defaults to ''.
        embeddings_path (str, optional): Path to save the results in. Defaults to ''.

    Returns:
        embeddings (torch.Tensor): embeddings from the transformer model
    """
    snapshot_path = Path(snapshot_path)
    embeddings_path = Path(embeddings_path) / f'{embedder}.pkl'
    embeddings_path.parent.mkdir(exist_ok=True,parents=True)
    whole_text = get_snapshot_column(snapshot_path,'whole_text')
        
    
    if embeddings_path.is_file():
        pickle.load(open(embeddings_path,'rb'))
    
    else:
        embeddings = []
        for batch_start in tqdm(
            range(0,len(whole_text,batch_size)),
            desc='creating embeddings',
            # leave=False
        ):  
            batch_embeddings = batch_transformer_embeddings(whole_text[batch_start:batch_start+batch_size],embedder)
            embeddings.append(batch_embeddings)        
            torch.cuda.empty_cache()
            
        embeddings = torch.vstack(embeddings)
        torch.save(embeddings,embeddings_path)
    return embeddings