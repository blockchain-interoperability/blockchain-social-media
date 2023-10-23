import networkx as nx
import json
import time

from crypto_chatter.utils import progress_bar
from crypto_chatter.graph import CryptoGraph

def load_graph_components(
    graph: CryptoGraph,
    top_n: int = 100,
) -> list[list[int]]:
    '''
    Loads the strongly connected components of the given directed graph.
    '''
    marker_file = graph.data_config.graph_components_dir/'completed.txt'
    last_file = sorted(graph.data_config.graph_components_dir.glob('*.json'))[top_n-1]
    if not marker_file.is_file() or not last_file.is_file():
        start = time.time()
        connected_components = [
            list(cc) 
            for cc in sorted(nx.strongly_connected_components(graph.G), key=len, reverse=True)
        ]
        print(f'detected {len(connected_components):,} components in {int(time.time()-start)} seconds')
        with progress_bar() as progress:
            save_task = progress.add_task(
                description='saving component info..', 
                total=max(len(connected_components), top_n),
            )
            for i, cc in enumerate(connected_components[:top_n]):
                json.dump(
                    cc,
                    open(
                        graph.data_config.graph_components_dir / f'{i:06}.json',
                        'w'
                    )
                )
                progress.update(save_task, advance =1)
        print(f'counted and saved connected components info in {int(time.time()-start)} seconds')
        open(marker_file, 'w').close()

    else:
        start = time.time()
        cc_files = sorted(graph.data_config.graph_components_dir.glob('*.json'))[:top_n]
        connected_components = []
        with progress_bar() as progress:
            load_task = progress.add_task(description='loading component info..', total=len(cc_files))
            for f in cc_files:
                connected_components += [json.load(open(f))]
                progress.update(load_task, advance =1)
        print(f'loaded {top_n} compnents in {int(time.time()-start)} seconds')
    
    return connected_components
