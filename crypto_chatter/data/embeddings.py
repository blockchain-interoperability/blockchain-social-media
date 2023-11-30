import torch
import torch.nn.functional as F
from rich.progress import Progress
import numpy as np
import math
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

def setup_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(f'sentence-transformers/{model_name}')
    model = AutoModel.from_pretrained(f'sentence-transformers/{model_name}').to(device)

    def generate(
        texts:list[str]
    ) -> np.ndarray:
        print(f'received {len(texts)} texts')
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            model_output = model(
                input_ids = encoded_input['input_ids'].to(device),
                token_type_ids = encoded_input['token_type_ids'].to(device),
                attention_mask = encoded_input['attention_mask'].to(device),
            )
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.detach().cpu().numpy()
    return generate, model, tokenizer

def get_sbert_embeddings(
    text: list[str]|np.ndarray,
    data_config: CryptoChatterDataConfig,
    ids: IdList,
    model_name:str = "all-MiniLM-L12-v2",
    batch_size: int = 256,
    progress: Progress|None = None,
) -> None:
    if device == 'cpu': batch_size = 128
    save_dir = data_config.data_dir / "embeddings" / model_name
    save_dir.mkdir(exist_ok=True,parents=True)

    all_embeddings = np.zeros((len(text), 384))
    save_files = [save_dir / f"{int(_id)}.npy" for _id in ids]
    incomplete_files, incomplete_idxs = [],[]

    for i, file in enumerate(save_files):
        if file.is_file():
            all_embeddings[i] = np.load(open(file, "rb"))
        else:
            incomplete_idxs += [i]
            incomplete_files += [file]

    if len(incomplete_idxs) > 0:
        num_iters = math.ceil(len(incomplete_idxs) / batch_size)

        progress_task = None
        if progress is not None:
            progress_task = progress.add_task(
                "generating embeddings..",
                total=num_iters,
            )

        generate, model, tokenizer = setup_model(model_name)

        for offset in range(0, len(incomplete_idxs), batch_size):
            batch_text = [
                preprocess_text(text[j]) 
                for j in incomplete_idxs[offset:offset+batch_size]
            ]
            new_embeddings = generate(texts=batch_text)

            for batch_idx, embedding in enumerate(new_embeddings):
                np.save(open(incomplete_files[offset+batch_idx], "wb"),embedding)
                all_embeddings[incomplete_idxs[offset+batch_idx]] = embedding

            if progress is not None:
                progress.advance(progress_task)

        if progress is not None:
            progress.remove_task(progress_task)

        del model
        del tokenizer

    if (all_embeddings == 0).all(axis=1).any():
        raise Exception("Some embeddings were not generated!")

    return all_embeddings
