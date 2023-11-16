import pandas as pd
import numpy as np
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import progress_bar, device

from .utils import preprocess_text

def generate_roberta_sentiment(
    text: list[str]|np.ndarray|pd.Series,
    data_config: CryptoChatterDataConfig,
    indices: list[int]|np.ndarray,
    model_name:str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> None:
    save_dir = data_config.data_dir / "sentiment" / model_name.replace("/", "_")
    save_dir.mkdir(exist_ok=True,parents=True)

    all_generated = all((save_dir/f'{int(i)}.json').is_file() for i in indices)

    if not all_generated:
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

        with progress_bar() as progress:
            analyze_task = progress.add_task(
                "analyzing sentiment..",
                total=len(text)
            )

            text_iterable = text
            if isinstance(text,pd.Series):
                text_iterable = text.values

            text_iterable = zip(indices, text_iterable)

            for i, one_text in text_iterable:
                save_file = save_dir / f"{int(i)}.json"
                if save_file.is_file(): continue
                sentiment = analyze(one_text)
                json.dump(sentiment, open(save_file, "w"))
                progress.advance(analyze_task)
    else:
        print("Sentiment analysis already completed. Skipping..")

def get_roberta_sentiment(
    row_index: int,
    data_config: CryptoChatterDataConfig,
    text: str = '',
    model_name:str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> torch.Tensor: 
    save_file = data_config.data_dir / "sentiment" / model_name / f"{int(row_index)}.json"

    if not save_file.is_file():
        generate_roberta_sentiment(
            text=[text],
            data_config=data_config,
            indices=[row_index],
            model_name=model_name,
        )
    return json.load(open(save_file))
