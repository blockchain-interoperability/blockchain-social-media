import pandas as pd
import numpy as np
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import progress_bar, device
from crypto_chatter.utils.types import (
    Sentiment,
    IdList,
)

from .utils import preprocess_text

def get_roberta_sentiments(
    text: list[str]|np.ndarray,
    data_config: CryptoChatterDataConfig,
    ids: IdList,
    model_name:str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> list[Sentiment]:
    save_dir = data_config.data_dir / "sentiment" / model_name.replace("/", "_")
    save_dir.mkdir(exist_ok=True,parents=True)

    sentiments = []
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
            sentiments += [sentiment]
            progress.advance(analyze_task)
            
    del model
    del tokenizer

    return sentiments
