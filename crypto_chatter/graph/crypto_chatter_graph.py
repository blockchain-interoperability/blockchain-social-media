from typing import Literal
from typing_extensions import Self
import time
import networkx as nx
import numpy as np
import pandas as pd
import json
from collections import Counter

from crypto_chatter.config import CryptoChatterDataConfig, CryptoChatterGraphConfig
from crypto_chatter.data import CryptoChatterData
from crypto_chatter.utils import NodeList, EdgeList

UndirectedNodeCentralityType = Literal['degree', 'closeness']

class CryptoChatterSubGraph:
    parent: 'CryptoChatterGraph'
    source: int
    nodes: NodeList
    graph: nx.Graph
    data: pd.DataFrame

    def __init__(
        self, 
        parent: 'CryptoChatterGraph', 
        source: int,
    ):
        self.parent = parent
        self.source = source
        self.nodes = self.parent.get_all_reachable_nodes(self.source)
        self.graph = self.parent.G.subgraph(self.nodes)
        self.data = CryptoChatterData(
            self.parent.data.data_config,
            df = self.parent.data[self.parent.data["id"].isin(self.nodes)]
        )

    def get_keywords(
        self,
        top_n: int = 100,
    ) -> dict[str, float]:
        save_file = self.parent.graph_config.graph_dir / f"subgraph/{self.source}/keywords/{self.parent.data.tfidf_settings}/{top_n}.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        if not save_file.is_file():
            if self.parent.data.tfidf is None:
                self.parent.data.fit_tfidf()
            terms = self.parent.data.tfidf.get_feature_names_out()
            vecs = self.parent.data.tfidf.transform(self.data.text)
            tfidf_scores = vecs.toarray().sum(0)
            sorted_idxs = tfidf_scores.argsort()[::-1]
            keywords = terms[sorted_idxs][:top_n]
            keyword_scores = tfidf_scores[sorted_idxs][:top_n]
            keywords_with_score = dict(zip(keywords, keyword_scores))
            json.dump(keywords_with_score, open(save_file, "w"))
        else:
            keywords_with_score = json.load(open(save_file))

        return keywords_with_score

    def count_hashtags(
        self,
        top_n:int = 100,
    ) -> dict[str, int]:
        save_file = self.parent.graph_config.graph_dir / f"subgraph/{self.source}/hashtags/{top_n}.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        if not save_file.is_file():
            hashtag_count = dict(
                Counter([
                    tag
                    for hashtags in self.data["hashtags"].values
                    for tag in hashtags
                ]).most_common()[:top_n]
            )
            json.dump(hashtag_count, open(save_file, "w"))
        else:
            hashtag_count = json.load(open(save_file))
        return hashtag_count

class CryptoChatterGraph:
    G: nx.DiGraph
    nodes: NodeList
    edges: EdgeList
    data: CryptoChatterData
    graph_config: CryptoChatterGraphConfig
    data_source: str
    top_n_components: int 
    components: list[NodeList] | None = None

    def __init__(
        self, 
        graph_config: CryptoChatterGraphConfig,
        data_config: CryptoChatterDataConfig,
    ) -> None:
        self.graph_config = graph_config
        self.build(data_config)
        
    def build(
        self,
        data_config:CryptoChatterDataConfig,
    ) -> None:
        ...

    def load_components(
        self,
    ) -> Self:
        ...

    def degree(
        self,
    ):
        save_file = self.graph_config.graph_dir / "stats/out_degree.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)

        if not save_file.is_file():
            start = time.time()
            degree = list(dict(self.G.degree(self.nodes)).values())
            print(f"computed degree stats in {int(time.time() - start)} seconds")
            json.dump(degree, open(save_file, "w"))
        else:
            degree = json.load(open(save_file))
        return np.array(degree)

    def degree_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.graph_config.graph_dir / "stats/degree_centrality.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)

        if not save_file.is_file():
            start = time.time()
            deg_cent = nx.degree_centrality(self.G)
            deg_cent_values = [deg_cent[n] for n in self.nodes]
            print(f"computed degree centrality in {int(time.time() - start)} seconds")
            json.dump(deg_cent_values, open(save_file,"w"))
        else:
            deg_cent_values = json.load(open(save_file))
        return np.array(deg_cent_values)

    def betweenness_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.graph_config.graph_dir / "stats/betweenness_centrality.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)

        if not save_file.is_file():
            start = time.time()
            bet_cent = nx.betweenness_centrality(self.G)
            bet_cent_values = [bet_cent[n] for n in self.nodes]
            print(f"computed betweenness centrality in {int(time.time() - start)} seconds")
            json.dump(bet_cent_values, open(save_file,"w"))
        else:
            bet_cent_values = json.load(open(save_file))
        return np.array(bet_cent_values)

    def eigenvector_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.graph_config.graph_dir / "stats/eigenvector_centrality.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)

        if not save_file.is_file():
            start = time.time()
            eig_cent = nx.eigenvector_centrality(self.G)
            eig_cent_values = [eig_cent[n] for n in self.nodes]
            print(f"computed eigenvector centrality in {int(time.time() - start)} seconds")
            json.dump(eig_cent_values, open(save_file,"w"))
        else:
            eig_cent_values = json.load(open(save_file))
        return np.array(eig_cent_values)

    def closeness_centrality(
        self,
    ) -> np.ndarray:
        save_file = self.graph_config.graph_dir / "stats/closeness_centrality.json"
        save_file.parent.mkdir(parents=True, exist_ok=True)

        if not save_file.is_file():
            start = time.time()
            cls_cent = nx.closeness_centrality(self.G)
            cls_cent_values = [cls_cent[n] for n in self.nodes]
            print(f"computed closeness centrality in {int(time.time() - start)} seconds")
            json.dump(cls_cent_values, open(save_file,"w"))
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

