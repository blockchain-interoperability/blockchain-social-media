from typing_extensions import Self
import time
import networkx as nx
import numpy as np
import pandas as pd
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from crypto_chatter.config import CryptoChatterDataConfig
from crypto_chatter.utils import NodeList, EdgeList

class CryptoGraph:
    G: nx.DiGraph
    nodes: NodeList
    edges: EdgeList
    data: pd.DataFrame
    data_config: CryptoChatterDataConfig
    node_id_col: str
    data_source: str
    top_n_components: int 
    components: list[NodeList] | None = None
    tfidf: TfidfVectorizer | None = None

    def __init__(self, data_config: CryptoChatterDataConfig) -> None:
        self.data_config = data_config
        self.build()
        
    def fit_tfidf(
        self,
        random_seed = 0,
        random_size = 1000000,
    ) -> Self:
        save_dir = self.data_config.graph_dir / f'stats/tfidf_{random_seed}_{random_size}.pkl'
        if not save_dir.is_file():
            start = time.time()
            rng = np.random.default_rng(random_seed)
            random_idxs = rng.permutation(np.arange(len(self.data)))[:random_size]
            subset = self.data[self.data_config.text_col].values[random_idxs]
            self.tfidf = TfidfVectorizer(stop_words='english')
            self.tfidf.fit(subset)
            pickle.dump(self.tfidf, open(save_dir,'wb'))
            print(f'computed tfidf and saved in {int(time.time() - start)} seconds')
        else:
            self.tfidf = pickle.load(open(save_dir, 'rb'))
        return self

    def build(self) -> None:
        ...

    def load_components(
        self,
    ) -> Self:
        ...

    def degree(
        self,
    ):
        save_file = self.data_config.graph_dir / 'stats/out_degree.json'
        if not save_file.is_file():
            start = time.time()
            degree = list(dict(self.G.degree(self.nodes)).values())
            print(f'computed degree stats in {int(time.time() - start)} seconds')
            json.dump(degree, open(save_file, 'w'))
        else:
            degree = json.load(open(save_file))
        return np.array(degree)

    def degree_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.data_config.graph_dir / 'stats/degree_centrality.json'
        if not save_file.is_file():
            start = time.time()
            deg_cent = nx.degree_centrality(self.G)
            deg_cent_values = [deg_cent[n] for n in self.nodes]
            print(f'computed degree centrality in {int(time.time() - start)} seconds')
            json.dump(deg_cent_values, open(save_file,'w'))
        else:
            deg_cent_values = json.load(open(save_file))
        return np.array(deg_cent_values)

    def betweenness_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.data_config.graph_dir / 'stats/betweenness_centrality.json'
        if not save_file.is_file():
            start = time.time()
            bet_cent = nx.betweenness_centrality(self.G)
            bet_cent_values = [bet_cent[n] for n in self.nodes]
            print(f'computed betweenness centrality in {int(time.time() - start)} seconds')
            json.dump(bet_cent_values, open(save_file,'w'))
        else:
            bet_cent_values = json.load(open(save_file))
        return np.array(bet_cent_values)

    def eigenvector_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.data_config.graph_dir / 'stats/eigenvector_centrality.json'
        if not save_file.is_file():
            start = time.time()
            eig_cent = nx.eigenvector_centrality(self.G)
            eig_cent_values = [eig_cent[n] for n in self.nodes]
            print(f'computed eigenvector centrality in {int(time.time() - start)} seconds')
            json.dump(eig_cent_values, open(save_file,'w'))
        else:
            eig_cent_values = json.load(open(save_file))
        return np.array(eig_cent_values)

    def closeness_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.data_config.graph_dir / 'stats/closeness_centrality.json'
        if not save_file.is_file():
            start = time.time()
            cls_cent = nx.closeness_centrality(self.G)
            cls_cent_values = [cls_cent[n] for n in self.nodes]
            print(f'computed closeness centrality in {int(time.time() - start)} seconds')
            json.dump(cls_cent_values, open(save_file,'w'))
        else:
            cls_cent_values = json.load(open(save_file))
        return np.array(cls_cent_values)

    def get_all_reachable_nodes(
        self, 
        node: int,
    ) -> NodeList:
        stack = [node]
        reachable = []
        while stack:
            current = stack.pop()
            reachable += [current]
            for neighbor in nx.all_neighbors(self.G, current):
                if neighbor not in reachable:
                    stack += [neighbor]
        return reachable

    def get_stats(
        self,
        recompute: bool = False,
        display: bool = False,
    ) -> dict[str, any]:
        ...

    def export_gephi_components(
        self,
    ) -> None:
        ...

class CryptoSubgraph:
    parent: CryptoGraph
    source: int
    nodes: NodeList
    graph: nx.Graph
    data: pd.DataFrame

    def __init__(
        self, 
        parent: CryptoGraph, 
        source: int,
    ):
        self.parent = parent
        self.source = source
        self.nodes = self.parent.get_all_reachable_nodes(self.source)
        self.graph = self.parent.G.subgraph(self.nodes)
        self.data = self.parent.data[self.parent.data.id.isin(self.nodes)]

    def get_tfidf(
        self,
    ):
        if self.parent.tfidf is None:
            self.parent = self.parent.fit_tfidf()
        vecs = self.parent.tfidf.transform(self.data[self.parent.data_config.text_col])


    def count_hashtags(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        hashtags = []
        counts = []
        hashtag_count = Counter([
            tag
            for hashtags in self.data['hashtags'].values
            for tag in hashtags
        ])
        if len(hashtag_count) > 0:
            hashtags, counts = zip(*hashtag_count.most_common())

        hashtags = np.array(hashtags)
        counts = np.array(counts)
        return hashtags, counts
