import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

def compute_aspect_similarities(
        tweet_embeddings: np.array, 
        embedding_type: str, 
        embedding_model: SentenceTransformer, 
        aspects: List[str]
    ):
    # Embed aspects
    if embedding_type == "sbert":
        aspect_embeddings = embedding_model.encode(aspects, normalize_embeddings=True)
    elif embedding_type == "use_large":
        aspect_embeddings = embedding_model(aspects).numpy()
    else:
        raise ValueError(f"Unsupported embedding type '{embedding_type}'.")

    # Compute aspect similarity vector for each response.
    # Matrix multiplication will give cosine similarities
    # since all embeddings are normalized to unit sphere.
    aspect_similarities = tweet_embeddings @ aspect_embeddings.T

    return aspect_similarities

