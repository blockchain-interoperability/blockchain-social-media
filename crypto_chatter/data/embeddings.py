from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import progress_bar,device
from crypto_chatter.utils.types import (
    IdList
)

from .utils import preprocess_text

def get_sbert_embeddings(
    text: list[str]|np.ndarray,
    data_config: CryptoChatterDataConfig,
    ids: IdList,
    model_name:str = "all-MiniLM-L12-v2",
) -> None:
    save_dir = data_config.data_dir / "embeddings" / model_name
    save_dir.mkdir(exist_ok=True,parents=True)

    embeddings = []
    model = SentenceTransformer(model_name, device=device)
    with progress_bar() as progress:
        generate_task = progress.add_task(
            "generating embeddings..",
            total=len(text)
        )

        for i, one_text in zip(ids, text):
            save_file = save_dir / f"{int(i)}.npy"
            if save_file.is_file(): 
                embedding = np.load(open(save_file, "rb"))
            else:
                embedding = model.encode(
                    preprocess_text(one_text),
                    convert_to_numpy=True,
                )
                np.save(open(save_file, "wb"),embedding)
            embeddings += [embedding]
            progress.advance(generate_task)

    del model

    return np.stack(embeddings)
