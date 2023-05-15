from typing import Tuple,List
import pandas as pd
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from textwrap import wrap
from sentence_transformers import SentenceTransformer

def text_wrap(text:str) -> str:
    return "<br>".join(wrap(text, width=80))

def get_tweet_text(hit:dict) -> Tuple[str,str]:
    text = (hit["extended_tweet"]["full_text"] if "extended_tweet" in hit 
            else hit["full_text"] if "full_text" in hit 
            else hit["text"])
    quoted_text = None
    if "quoted_status" in hit:
        quoted_status = hit["quoted_status"]
        quoted_text = (quoted_status["extended_tweet"]["full_text"] if "extended_tweet" in quoted_status 
                      else quoted_status["full_text"] if "full_text" in quoted_status 
                      else quoted_status["text"])

    return text, quoted_text

def get_base_filters(
        embedding_type:str, 
        use_responses:bool
    ) -> dict:
    
    if use_responses:
        return [{
            "exists": {
                "field": f"embedding.{embedding_type}.quoted"
            },
            "exists": {
                "field": "timestamp_ms"
            }
        }, {
            "exists": {
                "field": f"embedding.{embedding_type}.primary"
            }
        }]
    else:
        return [{
            "exists": {
                "field": f"embedding.{embedding_type}.primary"
            },
            "exists": {
                "field": "timestamp_ms"
            }
        }]

def get_index_date_boundaries(
        es_uri:str, 
        es_index:str, 
        embedding_type:str, 
        use_responses:bool
    ) -> Tuple[datetime, datetime]:
    with Elasticsearch(hosts=[es_uri], timeout=60, verify_certs=False) as es:
        s = Search(using=es, index=es_index)
        s = s.params(size=0)
        s.update_from_dict({
            "query": {
                "bool": {"filter": get_base_filters(embedding_type, use_responses)}
            },
            "aggs": {
                "min_date": {"min": {"field": "created_at", "format": "strict_date"}},
                "max_date": {"max": {"field": "created_at", "format": "strict_date"}}
            }
        })
        results = s.execute()
    min_date = datetime.strptime(results.aggregations.min_date.value_as_string, "%Y-%m-%d").date()
    max_date = datetime.strptime(results.aggregations.max_date.value_as_string, "%Y-%m-%d").date()
    return min_date, max_date

def get_query(
        embedding_type: str, 
        query_embedding: str, 
        date_range: Tuple[datetime,datetime], 
        sentiment_type: str,
        use_responses: bool,
    ) -> dict:
    additional_filters = []
    if len(date_range) > 0:
        additional_filters.append({
            "range": {
                "created_at": {
                    "format": "strict_date",
                    "time_zone": "+00:00",
                    "gte": date_range[0].strftime("%Y-%m-%d")
                }
            }
        })
        if len(date_range) > 1:
            additional_filters[-1]["range"]["created_at"]["lte"] = date_range[1].strftime("%Y-%m-%d")

    if use_responses:
        source_keys = [
            "id_str", "timestamp_ms", "text", "extended_tweet.full_text", "quoted_status.text", 
            "quoted_status.extended_tweet.full_text", f"embedding.{embedding_type}.primary",
            f"sentiment.{sentiment_type}.primary"
        ]
        match_function = f"dotProduct(params.query_vector, 'embedding.{embedding_type}.quoted') + 1.0"
    else:
        source_keys = [
            "id_str", "timestamp_ms", "text", "extended_tweet.full_text", f"embedding.{embedding_type}.primary",
            f"sentiment.{sentiment_type}.primary"
        ]
        match_function = f"dotProduct(params.query_vector, 'embedding.{embedding_type}.primary') + 1.0"


    query = {
        "_source": source_keys,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "filter": get_base_filters(embedding_type, use_responses) + additional_filters
                    }
                },
                "script": {
                    
                    "source": match_function,
                    "params": {"query_vector": query_embedding.tolist()}
                }
            }
        }
    }
    return query

def run_query(
        es_uri: str, 
        es_index: str, 
        embedding_type: str, 
        embedding_model: SentenceTransformer, 
        query: dict, 
        date_range: Tuple[datetime,datetime],
        sentiment_type: str, 
        max_results: int = 1000,
        use_responses: bool = False
    ) -> Tuple[List[str],List[str],np.array,np.array,np.array,pd.Series]:

    print('sentiment is this: ', sentiment_type)
    # Embed query
    if embedding_type == "sbert":
        query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    elif embedding_type == "use_large":
        query_embedding = embedding_model([query]).numpy()[0]
    else:
        raise ValueError(f"Unsupported embedding type '{embedding_type}'.")

    # Use query embeddings to get responses to similar tweets
    with Elasticsearch(hosts=[es_uri], timeout=600000, verify_certs=False) as es:
        s = Search(using=es, index=es_index)
        s = s.params(size=max_results)
        s.update_from_dict(get_query(embedding_type, query_embedding, date_range, sentiment_type, use_responses))

        tweet_text = []
        tweet_text_display = []
        tweet_embeddings = []
        tweet_scores = []
        tweet_sentiments = []
        timestamp = []
        for hit in s.execute():
            tweet_embeddings.append(np.array(hit["embedding"][embedding_type]["primary"]))
            text, quoted_text = get_tweet_text(hit)
            tweet_scores.append(hit.meta.score-1.0)
            tweet_sentiments.append(np.array(hit["sentiment"][sentiment_type]["primary"]))
            timestamp.append(hit['timestamp_ms'])
            if use_responses:
                tweet_text.append((quoted_text, text))
                tweet_text_display.append(
                    f"Tweet:<br>----------<br>{text_wrap(quoted_text)}<br><br>"
                    f"Response:<br>----------<br>{text_wrap(text)}"
                )
            else:
                tweet_text.append((text,))
                tweet_text_display.append(
                    f'Tweet:<br>----------<br>{text_wrap(text)}'
                    f'<br>----------<br>Sentiment Score: {tweet_sentiments[-1]:.4f}'
                    f'<br>----------<br>Timestamp: {str(pd.to_datetime(timestamp[-1],unit="ms"))}'
                    )
            if len(tweet_embeddings) == max_results:
                break
        
        # print(timestamp)
        timestamp = pd.to_datetime(timestamp,unit='ms')
        tweet_embeddings = np.vstack(tweet_embeddings)
        tweet_sentiments = np.hstack(tweet_sentiments)
        tweet_scores = np.array(tweet_scores)
    print(f'The query got {len(tweet_embeddings)} tweets')

    return tweet_text, tweet_text_display, tweet_embeddings, tweet_scores, tweet_sentiments, timestamp

