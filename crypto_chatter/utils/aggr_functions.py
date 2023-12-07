import numpy as np

def safemean(values):
    if len(values) == 0: return 0.0
    else: return float(np.mean(values))

aggr_funcs = {
    "first": lambda x: x[0],
    "last": lambda x: x[-1],
    "min": lambda x: int(np.min(x)),
    "max": lambda x: int(np.max(x)),
    "mean": safemean,
    "sum": lambda x: int(np.sum(x)),
}
