import networkx as nx
import numpy as np
import time

from crypto_chatter.graph import CryptoTwitterReplyGraph

graph = CryptoTwitterReplyGraph('blockchain-interoperability-attacks')
graph.load_components()

connected_components_size = np.array([len(cc) for cc in graph.components])

print('maximum connected_components size:', max(connected_components_size))

