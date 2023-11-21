from rich.progress import Progress
import numpy as np
from sentence_transformers import SentenceTransformer

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import device
from crypto_chatter.utils.types import (
    IdList
)

from .text import preprocess_text

def get_sbert_embeddings(
    text: list[str]|np.ndarray,
    data_config: CryptoChatterDataConfig,
    ids: IdList,
    model_name:str = "all-MiniLM-L12-v2",
    progress: Progress|None = None,
) -> None:
    save_dir = data_config.data_dir / "embeddings" / model_name
    save_dir.mkdir(exist_ok=True,parents=True)

    embeddings = []
    model = SentenceTransformer(model_name, device=device)

    progress_task = None
    if progress is not None:
        progress_task = progress.add_task(
            "generating embeddings..",
            total=len(text)
        )

    use_progress = progress is not None and progress_task is not None

    for _id, one_text in zip(ids, text):
        save_file = save_dir / f"{int(_id)}.npy"
        if save_file.is_file(): 
            embedding = np.load(open(save_file, "rb"))
        else:
            embedding = model.encode(
                preprocess_text(one_text),
                convert_to_numpy=True,
            )
            np.save(open(save_file, "wb"),embedding)
        embeddings += [embedding]
        if use_progress:
            progress.advance(progress_task)

    if use_progress:
        progress.remove_task(progress_task)

    del model

    return np.stack(embeddings)
