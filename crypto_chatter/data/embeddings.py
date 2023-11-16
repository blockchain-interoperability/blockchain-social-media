from sentence_transformers import SentenceTransformer
import pandas as pd
import torch

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import progress_bar,device

from .utils import preprocess_text

def generate_sbert_embeddings(
    text: pd.Series | list[str],
    data_config: CryptoChatterDataConfig,
    model_name:str = "all-MiniLM-L12-v2",
) -> None:
    save_dir = data_config.data_dir / "embeddings" / model_name
    save_dir.mkdir(exist_ok=True,parents=True)
    marker_file = save_dir / 'completed.txt'

    if not marker_file.is_file():
        model = SentenceTransformer(model_name, device=device)
        with progress_bar() as progress:
            generate_task = progress.add_task(
                "generating embeddings..",
                total=len(text)
            )
            text_iterable = text
            if isinstance(text,pd.Series):
                text_iterable = text.values
            for i, one_text in enumerate(text_iterable):
                save_file = save_dir / f"{i}.pkl"
                if save_file.is_file(): continue
                embedding = model.encode(
                    preprocess_text(one_text),
                )
                torch.save(embedding, open(save_file, "wb"))
                progress.advance(generate_task)
        del model
        marker_file.touch()
    else:
        print("Embeddings already generated. Skipping..")

def get_sbert_embedding(
    row_index: int,
    data_config: CryptoChatterDataConfig,
    model_name:str = "all-MiniLM-L12-v2",
) -> torch.Tensor: 
    save_file = data_config.data_dir / "embeddings" / model_name / f"{row_index}.pkl"

    if not save_file.is_file():
        raise Exception("Embeddings are not generated! Please run generation first..")

    return torch.load(open(save_file, "rb"))
