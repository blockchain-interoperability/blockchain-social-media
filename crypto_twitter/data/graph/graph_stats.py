import networkx as nx
import time

from .build_graph import build_graph

def graph_stats() -> None:
    start = time.time()
    nodes,edges = build_graph()
    G = nx.DiGraph(edges)

    start = time.time()
    longest_path = nx.dag_longest_path(G)
    print(f'found longest path in {int(time.time() - start)} seconds')

    start = time.time()
    in_degree = G.in_degree(G.nodes)
    print(f'computed in_degree in {int(time.time() - start)} seconds')

    start = time.time()
    out_degree = G.out_degree(G.nodes)
    print(f'computed out_degree in {int(time.time() - start)} seconds')

    print('='*60)
    print('Graph Stats: ')
    print(f'{len(nodes)} nodes, {len(edges)} edges.')
    print(f'Longest Path: {len(longest_path)} nodes.')
    print(f'In Degree: avg -- {sum(in_degree)/len(in_degree):.2f}, max -- {max(in_degree)}')
    print(f'Out Degree: avg -- {sum(out_degree)/len(out_degree):.2f}, max -- {max(out_degree)}')
    print('='*60)


