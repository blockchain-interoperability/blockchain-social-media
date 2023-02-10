# Utility for interacting with elasticsearch

Credits: [twitter-nlp](https://github.com/TheRensselaerIDEA/twitter-nlp) by Abraham Sanders

## Workflow for creating a subset of an index

1. Create empty index

2. Run reindexing to copy over randomly selected set of random subset

3. Run embedder to add SBERT embeddings to the said index




## Using the streamlit dashboard

The dashboard can be ran after following the steps
1. Clone this repository in your local machine
2. Recreate the blockchain-sns-env by creating a python 3.10 venv and running `pip install -r requirements.txt`
3. Open ssh port-forwarding to idea cluster
4. Move to `tools/apsect_modeling`
5. Run `streamlit run demo_streamlit.py`

*Note: you may run into a timeout error when loading the SBERT embedder model. To download the model ahead of time, open the python shell (make sure in venv) and run:*

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L12-v2')
```