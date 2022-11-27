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

from dataset import TwitterDataset
from autoencoders import load_encoder

def run_kmeans(
    timestamp_path,
    spam_idx_path,
    embedding_path,
    sentiment_path,
    token_path,
    encoder_path,
    encoder_type,
    k = 7,
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
    save_path = Path(save_path)/f'kmeans_{k}_{slice_size}_results.pkl'
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
    for batch in tqdm(loader):
        inp = batch['embedding'].cuda()
        out = model.encoder(inp)
        reduced.append(out)
        # break
    reduced = torch.vstack(reduced)

    scaled = MaxAbsScaler().fit_transform(reduced.cpu().numpy())
    cluster_ids_x, cluster_centers = kmeans(
        X=scaled, num_clusters=k, distance='euclidean', device=torch.device('cuda:0'), iter_limit =100
    )
    print('clusters identified')


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
    results = []
    for slice_beg,slice_end in tqdm(time_slices,desc=f'going over by {slice_size}'):
        # grab the dataset's sorted indexs using the time slices
        beg,end = dset.get_range(slice_beg,slice_end)
        
        
        labels = cluster_ids_x[beg:end]

        one_slice = []

        for label in set(labels):
            centroid = cluster_centers[label]
            cluster = scaled[beg:end][labels == label]
            distance = np.sqrt(((cluster - centroid)**2).sum(1))
            distance /= distance.max()
            original_index = dset[beg:end]['original_index'][labels == label]
            # idx_by_distance = np.arange(beg,end)[]
            # idx_by_distance = np.argsort(distance)


            word_count = {}
            word_weighted = {}
            sentiment_weighted = {0:0, 1:0, -1:0}
            sentiment_count = {0:0, 1:0, -1:0}

            for idx,dis in zip(original_index,distance):
                # word_counts += 
                for t in dset.tokens[idx]:
                    if not t in word_count:
                        word_count[t] = 1
                    else:
                        word_count[t] += 1
                    if not t in word_weighted:
                        word_weighted[t] = dis
                    else:
                        word_weighted[t] += dis
                
                sentiment_count[dset.sentiment_label[idx]] += 1
                sentiment_weighted[dset.sentiment_label[idx]] += dis

            one_slice.append({
                'topic_weighted': word_weighted,
                'topic_count': word_count,
                'sentiment_count': sentiment_count,
                'sentiment_weighted': sentiment_weighted,
            })
        results.append(one_slice)

    torch.save(results,save_path)
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
