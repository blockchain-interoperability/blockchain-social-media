from typing import List
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def get_time_states(timestamp,subgroup):
    if len(subgroup):
        pos_ratio = (subgroup['sentiment'] > .1).sum() / len(subgroup)
        neg_ratio = (subgroup['sentiment'] < -.1).sum() / len(subgroup)
        neu_ratio = 1 - pos_ratio - neg_ratio
        # non_ratio = 0
    else:
        pos_ratio = 0
        neg_ratio = 0
        neu_ratio = 0
        # non_ratio = 1
   
    return {
        'timestamp':timestamp,
        'tweet_count': len(subgroup),
        'pos_ratio': pos_ratio,
        'neg_ratio': neg_ratio,
        'neu_ratio': neu_ratio,
        # 'non_ratio': non_ratio,
        'average': (subgroup['sentiment'].mean() + 1)/2,
    }

def plot_cluster_sentiment(
        cluster_assignments:np.array,
        tweet_sentiments:pd.Series,
        timestamp:List[int],
        cluster_id:int = None
):

    # timestamp = pd.to_datetime(timestamp,unit='ms')
    df = pd.DataFrame(data = {'timestamp': timestamp, 'sentiment': tweet_sentiments, 'cluster': cluster_assignments})

    if cluster_id != None: 
        df = df[df['cluster'] == cluster_id]

    per_delta = pd.DataFrame([
        get_time_states(timestamp,subgroup)
        for timestamp,subgroup in df.resample(f'{6*60}min', on='timestamp')
    ])

    count_trace = go.Scatter(
        name = 'Tweet Count',
        x = per_delta['timestamp'],
        y = per_delta['tweet_count']
    )

    pos_trace = go.Bar(
        name = 'Positive Ratio',
        x = per_delta['timestamp'],
        y = per_delta['pos_ratio'],
        marker_color = '#6b95e8',
        xaxis = 'x2',
        yaxis = 'y2',
    )

    neg_trace = go.Bar(
        name = 'Negative Ratio',
        x = per_delta['timestamp'],
        y = per_delta['neg_ratio'],
        marker_color = '#e8776b',
        xaxis = 'x2',
        yaxis = 'y2',
    )

    neu_trace = go.Bar(
        name = 'Neutral Ratio',
        x = per_delta['timestamp'],
        y = per_delta['neu_ratio'],
        marker_color = '#e6e6e6',
        xaxis = 'x2',
        yaxis = 'y2',
    )

    avg_trace = go.Scatter(
        name = 'Average Sentiment',
        x = per_delta['timestamp'],
        y = per_delta['average'],
        line_color = '#44c767',
        xaxis = 'x2',
        yaxis = 'y2',
    )

    return go.Figure(
        data = [
            count_trace, avg_trace, neg_trace,neu_trace,pos_trace
        ],
        
        layout = {
            'xaxis': {'domain': [0,.4]},
            'xaxis2': {'domain': [.5,1]},
            'yaxis1': {'anchor': 'x2'},
            'barmode':'stack',
        }
    )
