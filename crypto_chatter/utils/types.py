from typing import Literal
import numpy as np
import pandas as pd

TwitterGraphKind = Literal['tweet','user']
GraphKind = Literal[TwitterGraphKind]

NodeList = list[int]
EdgeList = list[tuple[int,int]]
NodeToIndexMapping = dict[int,int]
IdList = list[int]
TextList = np.ndarray|list[str]|pd.Series

DirectedDegreeKind = Literal['in', 'out']
UndirectedDegreeKind = Literal['all']
DegreeKind = Literal[UndirectedDegreeKind, DirectedDegreeKind]

SentimentKind = Literal['negative', 'neutral', 'positive']
Sentiment = dict[SentimentKind, float]

DirectedCentralityKind = Literal['in_degree', 'out_degree']
UndirectedCentralityKind = Literal['degree', 'closeness', 'eigenvector']
CentralityKind = Literal[UndirectedCentralityKind, DirectedCentralityKind]

NodeAttributeList = list[str|float|int]|TextList
NodeAttributeDict = dict[int, str|float|int]
NodeAttributeKind = Literal['text','sentiment']

EdgeAttributeList = list[str|float|int]
EdgeAttributeDict = dict[int, str|float|int]
EdgeAttributeKind = Literal['emb_cosine_sim']
