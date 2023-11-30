import torch
import torch.nn.functional as F
from rich.progress import Progress
import numpy as np
# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import device
from crypto_chatter.utils.types import (
    IdList
)

from .text import preprocess_text

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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

    tokenizer = AutoTokenizer.from_pretrained(f'sentence-transformers/{model_name}')
    model = AutoModel.from_pretrained(f'sentence-transformers/{model_name}').to(device)

    def generate(
        text:str
    ) -> np.ndarray:
        encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(
                input_ids = encoded_input['input_ids'].to(device),
                token_type_ids = encoded_input['token_type_ids'].to(device),
                attention_mask = encoded_input['attention_mask'].to(device),
            )
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].detach().cpu().numpy()

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
            embedding = generate(
                preprocess_text(one_text),
            )
            np.save(open(save_file, "wb"),embedding)
        embeddings += [embedding]
        if use_progress:
            progress.advance(progress_task)

    if use_progress:
        progress.remove_task(progress_task)

    del model

    return np.stack(embeddings)
