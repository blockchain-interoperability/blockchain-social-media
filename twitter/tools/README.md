# Utility for interacting with elasticsearch

Credits: [twitter-nlp](https://github.com/TheRensselaerIDEA/twitter-nlp) by Abraham Sanders

## Workflow for creating a subset of an index

1. Create empty index

2. Run reindexing to copy over randomly selected set of random subset

3. Run embedder to add SBERT embeddings to the said index