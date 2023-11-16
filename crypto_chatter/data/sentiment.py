import pandas as pd
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
    text: pd.Series | list[str],
    data_config: CryptoChatterDataConfig,
    model_name:str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> None:
    save_dir = data_config.data_dir / "sentiment" / model_name.replace("/", "_")
    save_dir.mkdir(exist_ok=True,parents=True)
    marker_file = save_dir / 'completed.txt'

    if not marker_file.is_file():

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
            for i, one_text in enumerate(text_iterable):
                save_file = save_dir / f"{i}.json"
                if save_file.is_file(): continue
                sentiments = analyze(one_text)
                json.dump(sentiments, open(save_file, "w"))
                progress.advance(analyze_task)
        del model
        marker_file.touch()
    else:
        print("Sentiment analysis already completed. Skipping..")

def get_roberta_sentiment(
    row_index: int,
    data_config: CryptoChatterDataConfig,
    model_name:str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> torch.Tensor: 
    save_file = data_config.data_dir / "sentiment" / model_name / f"{row_index}.json"

    if not save_file.is_file():
        raise Exception("Sentiments are not generated! Please run generation first..")

    return json.load(open(save_file))
