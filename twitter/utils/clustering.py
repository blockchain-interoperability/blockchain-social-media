import torch
import pandas as pd
import pickle
# from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
# from sklearn.cluster import MiniBatchKMeans
from kmeans_pytorch.__init__ import kmeans
from torch.utils.data import DataLoader

from collections import Counter
from itertools import chain
from tqdm.auto import tqdm
from pathlib import Path

from utils.dataset import TwitterDataset
from utils.autoencoders import load_encoder

SENTIMENT_MAPPING = {
    -1: 'neg',
    0: 'neu',
    1: 'pos'
}

def run_kmeans(
    timestamp_path,
    spam_idx_path,
    embedding_path,
    sentiment_path,
    token_path,
    encoder_path,
    encoder_type,
    k = 7,
    top_n_topics = 20,
    slice_size = '30m',
    save_path = ''
):

    dset = TwitterDataset(
        timestamp_path,
        spam_idx_path,
        embedding_path = embedding_path,
        sentiment_path = sentiment_path,
        token_path = token_path,
    )
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True,parents=True)

    model = load_encoder(encoder_path,encoder_type)
    for p in model.parameters():
        p.requires_grad = False
    model = model.cuda()
    print('successfully loaded dataset and encoder')

    # first we establish the start time of the dataset
    dset_start = pd.to_datetime(dset.timestamp[dset.sorted_idx][0])
    dset_end = pd.to_datetime(dset.timestamp[dset.sorted_idx][-1])

    print(f'our dataset goes from {dset_start} to {dset_end}')

    loader = DataLoader(dset,batch_size=4096)
    reduced = []
    for batch in tqdm(loader,desc='reducing dimension...',leave=False):
        inp = batch['embedding'].cuda()
        out = model.encoder(inp)
        reduced.append(out)
        # break
    reduced = torch.vstack(reduced)

    scaled = MaxAbsScaler().fit_transform(reduced.cpu().numpy())
    cluster_ids_x, cluster_centers = kmeans(
        X=torch.tensor(scaled), num_clusters=k, distance='euclidean', device=torch.device('cuda:0'), iter_limit =100
    )
    print('clusters identified')

    cluster_ids_x = cluster_ids_x.numpy()
    cluster_centers = cluster_centers.numpy()
    # pre calculate the time slices acccording to the parameter
    time_slices = [
        [
            dset_start + pd.Timedelta(slice_size) * i, 
            dset_start + pd.Timedelta(slice_size) * (i+1)
        ]
        for i in range(
            int(np.ceil((dset_end - dset_start)/pd.Timedelta(slice_size)))-1
        )
    ]

    # start = dset_start
    embeddings = []
    results = []
    for i,(slice_beg,slice_end) in enumerate(tqdm(time_slices,desc=f'going over by {slice_size}')):
        # grab the dataset's sorted indexs using the time slices
        beg,end = dset.get_range(slice_beg,slice_end)
        
        
        labels = cluster_ids_x[beg:end]

        # one_slice = []

        for label in set(labels):
            centroid = cluster_centers[label]
            cluster = scaled[beg:end][labels == label]
            distance = np.sqrt(((cluster - centroid)**2).sum(1))
            distance /= distance.max()
            # higher is better. distance is normalized in [0,1] and by negating we give low distance a high weight
            weight = 1-distance 

            original_index = dset[beg:end]['original_index'][labels == label]
           # idx_by_distance = np.arange(beg,end)[]
            # idx_by_distance = np.argsort(distance)


            topic_count = {}
            topic_weighted = {}
            sentiment_weighted = {0:0, 1:0, -1:0}
            sentiment_count = {0:0, 1:0, -1:0}

            for idx,w in zip(original_index,weight):
                # word_counts += 
                for t in dset.tokens[idx]:
                    if not t in topic_count:
                        topic_count[t] = 1
                    else:
                        topic_count[t] += 1
                    if not t in topic_weighted:
                        topic_weighted[t] = w
                    else:
                        topic_weighted[t] += w
                
                sentiment_count[dset.sentiment_label[idx]] += 1
                sentiment_weighted[dset.sentiment_label[idx]] += w

            topic_by_weight,topic_weights = zip(*sorted(topic_weighted.items(),key=lambda x:x[1],reverse=True)[:top_n_topics])
            topic_by_count, topic_counts = zip(*sorted(topic_count.items(),key=lambda x:x[1],reverse=True)[:top_n_topics])

            sentiment_by_count = {f'{SENTIMENT_MAPPING[k]}_count': v for k,v in sentiment_count.items()}
            sentiment_by_weight = {f'{SENTIMENT_MAPPING[k]}_weight': v for k,v in sentiment_weighted.items()}

            results.append({
                'slice_idx': i,
                'label': label,
                'cluster_size': len(cluster),
                'topic_by_weight': topic_by_weight,
                'topic_weights': topic_weights,
                'topic_by_count': topic_by_count,
                'topic_counts': topic_counts,
                'original_idxs': original_index,
                **sentiment_by_count,
                **sentiment_by_weight,
            })
            embeddings.append({
                'slice_idx': i,
                'label': label,
                'original_idxs': original_index,
                'cluster_embs': cluster
            })

        # results.append(one_slice)

    # torch.save(results,save_path)
    df = pd.DataFrame(results)

    df.to_pickle(save_path/f'kmeans_{k}_{slice_size}_results.pkl')
    torch.save(embeddings,save_path/f'kmeans_{k}_{slice_size}_embeddings.pkl')
    # del original_embs,reduced_embs,scaled_embs


# def parse_clusters(
#     cluster_path,
#     timestamp_path,
#     spam_idx_path,
#     sentiment_path,
#     token_path,
# ):
#     cluster_path = Path(cluster_path)
#     cluster_files = sorted(cluster_path.glob('*.pkl'))
#     dset = TwitterDataset(
#         timestamp_path,
#         spam_idx_path,
#         sentiment_path=sentiment_path,
#         # whole_text_path = 'whole_text',
#         token_path = token_path,
#         # embedding_path = 'embeddings/all-MiniLM-L6-v2/',
#     )

#     all_clus_data = []
#     for one_slice in tqdm(cluster_files,leave=True,desc='iterating slices..'): 
#         cluster_data  = torch.load(one_slice)
#         slice_data = []
#         for one_cluster in cluster_data: 
#             word_counts = {}
#             sentiment_weighted = {0:0, 1:0, -1:0}
#             sentiment_count = {0:0, 1:0, -1:0}
#             for idx,dis in zip(one_cluster['idx_by_distance'],one_cluster['distance']): 
                
#                 for t in dset.tokens[idx]:
#                     if not t in word_counts:
#                         word_counts[t] = dis
#                     else:
#                         word_counts[t] += dis

#                 sentiment_weighted[dset.sentiment_label[idx]] += dis * dset.sentiment_score[idx]
#                 sentiment_count[dset.sentiment_label[idx]] += 1
#             words,scores = zip(*sorted(word_counts.items(),key=lambda x: x[1],reverse=True))
#             slice_data.append({
#                 'topics': words,
#                 'sentiment_weighted': sentiment_weighted,
#                 'sentiment_count': sentiment_count
#             })
#         all_clus_data.append(slice_data)
#         del one_slice

#     torch.save(all_clus_data,cluster_path/'parsed.pkl')
