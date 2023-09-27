import torch
from utils.autoencoders import LinearAutoEncoder
from utils.embeddings import get_sbert_embeddings
from pathlib import Path
from sklearn.cluster import KMeans
import time
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import json

def generateElbow(embeddings):
    #find elbow using inertia and distortion
    #use minibatch for optimal k computation to speed up calculations
    cluster_count = range(2, 21)
    batched_elbows = [0]*len(cluster_count)
    km_multiple = [KMeans(n_clusters=i, max_iter=100, n_init='auto') for i in cluster_count]
    size = 1000000
    it_count = 0
    for i in range(len(embeddings)):
        it_count+=1
        start = time.perf_counter()

        #batch testing
        batch = embeddings[i*size:(i+1)*size]
        if len(batch) == 0:
            break
        
        fits = [i.fit(batch.detach().numpy()) for i in km_multiple]
        inertias = [i.inertia_ for i in fits]
        batched_elbows = [a+b for a, b in zip(inertias, batched_elbows)]

        print("Iteration", it_count, f"Time taken:  {time.perf_counter()-start} seconds")

    knee_loc = KneeLocator(range(2, 21), batched_elbows, curve="convex", direction="decreasing")

    results = [x/it_count for x in batched_elbows]
    plt.figure(figsize=(6, 3.5))
    plt.plot(cluster_count, results)
    plt.xlim(2, 20)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average Inertia")
    plt.axvline(x=knee_loc.elbow, color="grey", linestyle='--')
    plt.savefig('kmeans_elbow.pdf')
    return knee_loc.elbow
    #plt.savefig('../../data/blockchain-interoperability/blockchain-social-media/twitter-data/kmeans_elbow.pdf')

def main():
    #load data
    DATA_DIR = Path('../twitter-data')

    embeddings = get_sbert_embeddings(
        snapshot_path = DATA_DIR/'snapshots',
        embeddings_path = DATA_DIR/'embeddings',
    )
    # and the model
    autoenc = LinearAutoEncoder()
    autoenc.load_state_dict(torch.load(DATA_DIR/'autoenc_10_epoch.pkl'))
    # freeze the model, because we don't want to do any more calculations. This saves computational power
    autoenc.requires_grad = False

    reduced_embs = autoenc.encoder(embeddings)
    model = autoenc.cuda()
    k = generateElbow(reduced_embs)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(reduced_embs.detach().numpy())

    indices={}
    for i in range(k):
        index_i = np.where(kmeans.labels_==i)[0]
        indices[i] = index_i.tolist()
        print(len(index_i))

    with open(DATA_DIR/'kmeans_clusters/kmeans_init_clusters.json', 'w') as f:
        json.dump(indices, f)

if __name__ == "__main__":
    main()