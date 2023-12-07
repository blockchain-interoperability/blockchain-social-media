from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import CryptoChatterGraphBuilder
from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig
from crypto_chatter.config.path import FIGS_DIR
from crypto_chatter.utils import progress_bar

import time
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns


dataset = "twitter:blockchain-interoperability-attacks"
graph_type = "tweet-quote"

data_config = CryptoChatterDataConfig(dataset)
graph_config = CryptoChatterGraphConfig(data_config, graph_type)

data = CryptoChatterData(
    data_config=data_config,
    columns=["hashtags"],
)
