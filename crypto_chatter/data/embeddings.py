from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import progress_bar,device

from .utils import preprocess_text

def generate_sbert_embeddings(
    text: list[str]|np.ndarray|pd.Series,
    data_config: CryptoChatterDataConfig,
    indices: list[int]|np.ndarray,
    model_name:str = "all-MiniLM-L12-v2",
) -> None:
    save_dir = data_config.data_dir / "embeddings" / model_name
    save_dir.mkdir(exist_ok=True,parents=True)

    all_generated = all((save_dir/f'{int(i)}.pkl').is_file() for i in indices)

    if not all_generated:
        model = SentenceTransformer(model_name, device=device)
        with progress_bar() as progress:
            generate_task = progress.add_task(
                "generating embeddings..",
                total=len(text)
            )

            text_iterable = text
            if isinstance(text,pd.Series):
                text_iterable = text.values

            text_iterable = zip(indices, text_iterable)

            for i, one_text in text_iterable:
                save_file = save_dir / f"{int(i)}.pkl"
                if save_file.is_file(): continue
                embedding = model.encode(
                    preprocess_text(one_text),
                    convert_to_numpy=True,
                )
                torch.save(embedding, open(save_file, "wb"))
                progress.advance(generate_task)
    else:
        print("Embeddings already generated. Skipping..")

def get_sbert_embedding(
    row_index: int,
    data_config: CryptoChatterDataConfig,
    text: str|None = None,
    model_name:str = "all-MiniLM-L12-v2",
) -> torch.Tensor: 
    save_file = data_config.data_dir / "embeddings" / model_name / f"{int(row_index)}.pkl"

    if not save_file.is_file():
        generate_sbert_embeddings(
            text=[text],
            data_config=data_config,
            indices=[row_index],
            model_name=model_name,
        )

    return torch.load(open(save_file, "rb"))
