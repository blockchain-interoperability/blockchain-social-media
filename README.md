# Blockchain Social Media


## Requirements

Python version: 3.10.12

Some pacakges require apt or conda to be installed.

[environment.yml](environment.yml)
[requirements.txt](requirements.txt)

To update the requirements file, run 

```bash
pip list --format=freeze > requirements.txt
```


To update the conda environment file, run
```bash
conda env export > environment.yml
```

## How to use:

### `.env` file setup

The .env file must be populated with the following variables:
```bash
ES_HOSTNAME={ADDRESS_TO_ELASTICSEARCH}
DATA_DIR={ABSOLUTE_PATH_TO_DATA_DIR}
```
The code will use the `.env` file to load the environment variables.


### Elasticsearch setup
The settings must be set in the config files.
An example config can be found in [./config/twitter:blockchain-interoperability-attacks](./config/twitter:blockchain-interoperability-attacks) 
```python
from crypto_chatter.data import CryptoChatterData
from crypto_chatter.graph import CryptoChatterGraph
from crypto_chatter.config import (
    CryptoChatterDataConfig,
    CryptoChatterGraphConfig
)

dataset = 'twitter:blockchain-interoperability-attacks'
graph_type = 'tweet'
data_config = CryptoChatterDataConfig(dataset)
graph_config = CryptoChatterGraphConfig(data_config, graph_type)
data = CryptoChatterData(
    data_config,
    columns = ['hashtags'],
)

graph = CryptoChatterGraph(
    data=data,
    graph_config=graph_config,
)
```
