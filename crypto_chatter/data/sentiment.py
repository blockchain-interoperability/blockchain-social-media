import numpy as np
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

    def to_list(self):
        return [
            self.positive,
            self.negative,
            self.neutral,
        ]

def get_roberta_sentiments(
    text: list[str]|np.ndarray,
    data_config: CryptoChatterDataConfig,
    ids: IdList,
    model_name:str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    progress: Progress|None = None,
) -> list[Sentiment]:
    save_dir = data_config.data_dir / "sentiment" / model_name.replace("/", "_")
    save_dir.mkdir(exist_ok=True,parents=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    def analyze(text:str):
        # TODO: fix the cuda integration for this.. not going to use it most likely though
        tokenized_text = tokenizer(
            preprocess_text(text), 
            return_tensors="pt"
        )
        logits = model(**tokenized_text)[0][0].softmax(dim=0).cpu()
        return {
            label: logits[_id].item()
            for _id, label in config.id2label.items()
        }

    progress_task = None
    if progress is not None:
        progress_task = progress.add_task(
            description="analyzing sentiment..",
            total=len(text)
        )

    use_progress = progress is not None and progress_task is not None
    sentiments = []

    for i, one_text in zip(ids, text):
        save_file = save_dir / f"{int(i)}.json"
        if save_file.is_file():
            sentiment = json.load(open(save_file))
        else:
            try:
                sentiment = analyze(one_text)
            except Exception as e:
                print(e)
                print(i)
                print(one_text)
                raise e

            json.dump(sentiment, open(save_file, "w"))
        sentiments += [Sentiment(**sentiment)]
        if use_progress:
            progress.advance(progress_task)

    if use_progress:
        progress.remove_task(progress_task)

    del model
    del tokenizer

    return sentiments
