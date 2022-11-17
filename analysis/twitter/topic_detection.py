from embeddings import get_bert_embeddings
def find_clusters(
    embedder ="all-MiniLM-L6-v2",
    cluster_algorithm="",
    
):
    embeddings = get_bert_embeddings(embedder);
    return
