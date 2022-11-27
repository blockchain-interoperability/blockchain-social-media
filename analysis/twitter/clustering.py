import torch
import pandas as pd
import pickle
# from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from sklearn.cluster import MiniBatchKMeans
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
    encoder_path,
    encoder_type,
    k = 7,
    slice_size = '30m',
    save_path = ''
):

    dset = TwitterDataset(
        timestamp_path,
        spam_idx_path,
        # sentiment_path,
        # whole_text_path = whole_text_path,
        # token_path =  token_path,
        embedding_path = embedding_path,
    )# torch.save(model.state_dict(),)
    
    save_path = Path(save_path)/f'kmeans_{k}_{slice_size}'
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
    for i,(slice_beg,slice_end) in enumerate(tqdm(time_slices,desc=f'going over by {slice_size}')):
        # grab the dataset's sorted indexs using the time slices
        beg,end = dset.get_range(slice_beg,slice_end)
        original_idxs = dset[beg:end]['original_index']
        original_embs = dset[beg:end]['embedding'].cuda()
        reduced_embs = model.encoder(original_embs).cpu().numpy()
        scaled_embs = MaxAbsScaler().fit_transform(reduced_embs)

        kmeans = MiniBatchKMeans(n_clusters=k).fit(scaled_embs)

        one_slice_result = []
        for label in set(kmeans.labels_):
            centroid = kmeans.cluster_centers_[label]
            cluster = scaled_embs[kmeans.labels_ == label]
            distance = np.sqrt(((cluster - centroid)**2).sum(1))
            distance /= distance.max()
            by_distance = original_idxs[kmeans.labels_ == label][np.argsort(distance)]
            one_slice_result.append({
                'time_start': slice_beg,
                'time_end': slice_end,
                'label': label,
                'by_distance': by_distance,
                'distance': distance,
                # 'cluster_values': cluster
                # 'word_count': word_count,
            })
        torch.save(one_slice_result,save_path/f'result_{i:03d}.pkl')
        torch.save(cluster,save_path/f'cluster_{i:03d}.pkl')
        # results.append(one_slice_result)
        
        del original_embs,reduced_embs,scaled_embs
        # results.append(one_slice_result)

        
    # pickle.dump(results,open(save_path/f'kmeans_{k}_{slice_size}.pkl','wb'))
