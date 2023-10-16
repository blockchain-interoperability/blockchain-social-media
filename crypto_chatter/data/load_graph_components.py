import networkx as nx
import json
import time

from crypto_chatter.config import CryptoChatterDataConfig

def load_graph_components(
    G: nx.DiGraph,
    data_config: CryptoChatterDataConfig,
) -> list[list[int]]:
    '''
    Loads the strongly connected components of the given directed graph.
    '''
    marker_file = data_config.graph_components_dir/'completed.txt'
    if not marker_file.is_file():
        start = time.time()
        connected_components = [list(cc) for cc in nx.strongly_connected_components(G)]
        for i, cc in enumerate(connected_components):
            json.dump(
                cc,
                open(
                    data_config.graph_components_dir / f'{i:06}.json',
                    'w'
                )
            )
        print(f'counted and saved connected components info in {int(time.time()-start)} seconds')
        open(marker_file, 'w').close()

    else:
        connected_components = [
            json.load(open(f))
            for f in sorted(data_config.graph_components_dir.glob('*.json'))
        ]
    
    return connected_components
