from typing import Literal
import numpy as np
import pandas as pd

TwitterGraphKind = Literal["tweet-quote", "tweet-reply", "user-quote", "user-reply"]
GraphKind = Literal[TwitterGraphKind]

NodeList = list[int] | np.ndarray
EdgeList = list[tuple[int, int]] | list[list[int]]
NodeToIdMap = dict[int, NodeList]
IdList = list[int] | np.ndarray
TextList = np.ndarray | list[str] | pd.Series

DirectedDegreeKind = Literal["in", "out"]
UndirectedDegreeKind = Literal["all"]
DegreeKind = Literal[UndirectedDegreeKind, DirectedDegreeKind]
DiameterKind = Literal["directed", "undirected"]

SentimentKind = Literal["negative", "neutral", "positive", "composite"]

DirectedCentralityKind = Literal["in_degree", "out_degree"]
UndirectedCentralityKind = Literal["degree", "closeness", "eigenvector"]
CentralityKind = Literal[UndirectedCentralityKind, DirectedCentralityKind]

SubGraphKind = Literal["centrality", "component", "community"]
ReachableKind = Literal["directed", "undirected", "reversed"]
ComponentKind = Literal["strong", "weak"]
CommunityKind = Literal["louvain"]
AttributeValues = list[str | float | int]

TweetGraphNodeAttributeKind = Literal[
    "text",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",
    "retweet_count",
    "favorite_count",
    "quote_count",
    "reply_count",
]
TweetAttributeKind = Literal[
    "text",
    "retweet_count",
    "favorite_count",
    "quote_count",
    "reply_count",
]
UserGraphNodeAttributeKind = Literal[
    "followers_count",
    "friends_count",
    "avg_retweet_count",
    "avg_quote_count",
    "avg_reply_count",
    "avg_favorite_count",
    "total_retweet_count",
    "total_quote_count",
    "total_reply_count",
    "total_favorite_count",
    "avg_sentiment_positive",
    "avg_sentiment_negative",
    "avg_sentiment_neutral",
]
UserAttributeKind = Literal[
    "user.followers_count",
    "user.friends_count",
    "reply_count",
    "favorite_count",
    "retweet_count",
    "quote_count",
]
AggrFunction = Literal["first", "last", "mean", "sum", "max", "min"]
NodeAttribute = dict[int, str | float | int]

TweetGraphEdgeAttributeKind = Literal["emb_cosine_sim"]
UserGraphEdgeAttributeKind = Literal["total_quote_count", "total_reply_count", "avg_quote_count", "avg_reply_count"]
UserInteractionKind = Literal["quote","reply"]
EdgeAttribute = dict[tuple[int, int], str | float | int]

ShortestPathKind = Literal["directed", "undirected", "reversed"]
