import numpy as np
import math
import json
from dataclasses import dataclass
from rich.progress import Progress
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    logging,
)

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import device
from crypto_chatter.utils.types import (
    IdList,
    SentimentKind,
)

from .text import preprocess_text

logging.set_verbosity_error()

@dataclass
class Sentiment:
    positive: float
    negative: float
    neutral: float

    def overall(self) -> str:
        return max([
            (self.positive, 'positive'), 
            (self.negative, 'negative'), 
            (self.neutral, 'neutral'),
        ])[1]

    def to_dict(self):
        return {
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
        }

    def __getitem__(
        self,
        key: SentimentKind,
    ) -> float:
        return self.to_dict()[key]

    def to_list(self):
        return [
            self.positive,
            self.negative,
            self.neutral,
        ]

def setup_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    def analyze(texts:str):
        # TODO: fix the cuda integration for this.. not going to use it most likely though
        tokenized_text = tokenizer(
            texts, 
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        logits = model(
            input_ids=tokenized_text.input_ids[:,:512].to(device),
            attention_mask=tokenized_text.attention_mask[:,:512].to(device),
        ).logits.softmax(dim=1).detach().cpu().numpy()
        return logits

    return analyze, model, tokenizer

def get_roberta_sentiments(
    text: list[str]|np.ndarray,
    data_config: CryptoChatterDataConfig,
    ids: IdList,
    model_name:str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 512,
    progress: Progress|None = None,
) -> list[Sentiment]:
    if device == 'cpu': batch_size = 128
    save_dir = data_config.data_dir / "sentiment" / model_name.replace("/", "_")
    save_dir.mkdir(exist_ok=True,parents=True)

    all_sentiments = np.zeros((len(text), 3))
    save_files = [save_dir / f"{int(i)}.npy" for i in ids]
    incomplete_idxs, incomplete_files = [], []

    for i, file in enumerate(save_files):
        if file.is_file():
            all_sentiments[i] = np.load(open(file, "rb"))
        else:
            incomplete_idxs += [i]
            incomplete_files += [file]

    if len(incomplete_idxs) > 0:
        num_iters = math.ceil(len(incomplete_idxs) / batch_size)

        progress_task = None
        if progress is not None:
            progress_task = progress.add_task(
                "analyzing sentiment..",
                total=num_iters,
            )

        analyze, model, tokenizer = setup_model(model_name)

        for offset in range(0, len(incomplete_idxs), batch_size):
            batch_text = [
                preprocess_text(text[j]) 
                for j in incomplete_idxs[offset:offset+batch_size]
            ]
            new_sentiments = analyze(texts=batch_text)

            for batch_idx, sentiment in enumerate(new_sentiments):
                np.save(open(incomplete_files[offset+batch_idx], "wb"),sentiment)
                all_sentiments[incomplete_idxs[offset+batch_idx]] = sentiment

            if progress is not None:
                progress.advance(progress_task)

        if progress is not None:
            progress.remove_task(progress_task)

        del model
        del tokenizer

    return [Sentiment(*s) for s in all_sentiments]
