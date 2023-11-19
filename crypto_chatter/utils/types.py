from typing import Literal
import numpy as np
import pandas as pd

TwitterGraphKind = Literal['tweet','user']
GraphKind = Literal[TwitterGraphKind]

NodeList = list[int]|np.ndarray
EdgeList = list[tuple[int,int]]|list[list[int]]
IdList = list[int]|np.ndarray
TextList = np.ndarray|list[str]|pd.Series

DirectedDegreeKind = Literal['in', 'out']
UndirectedDegreeKind = Literal['all']
DegreeKind = Literal[UndirectedDegreeKind, DirectedDegreeKind]

SentimentKind = Literal['negative', 'neutral', 'positive']
Sentiment = dict[SentimentKind, float]

DirectedCentralityKind = Literal['in_degree', 'out_degree']
UndirectedCentralityKind = Literal['degree', 'closeness', 'eigenvector']
CentralityKind = Literal[UndirectedCentralityKind, DirectedCentralityKind]

SubGraphKind = Literal['centrality', 'component']
ReachableKind = Literal['directed', 'undirected', 'reversed']

AttributeValues = list[str|float|int]

NodeAttributeKind = Literal['text','sentiment']
NodeAttribute = dict[int, str|float|int]

EdgeAttributeKind = Literal['emb_cosine_sim']
EdgeAttribute = dict[tuple[int,int], str|float|int]
