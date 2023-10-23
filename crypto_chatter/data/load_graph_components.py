import networkx as nx
import json
import time

from crypto_chatter.utils import progress_bar
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
        connected_components = [
            list(cc) 
            for cc in sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        ]
        print(f'detected {len(connected_components):,} components in {int(time.time()-start)} seconds')
        with progress_bar() as progress:
            save_task = progress.add_task(
                description='saving component info..', 
                total=len(connected_components)
            )

            for i, cc in enumerate(connected_components):
                json.dump(
                    cc,
                    open(
                        data_config.graph_components_dir / f'{i:06}.json',
                        'w'
                    )
                )
                progress.update(save_task, advance =1)
        print(f'counted and saved connected components info in {int(time.time()-start)} seconds')
        open(marker_file, 'w').close()

    else:
        cc_files = sorted(data_config.graph_components_dir.glob('*.json'))
        connected_components = []
        with progress_bar() as progress:
            load_task = progress.add_task(description='loading component info..', total=len(cc_files))
            for f in cc_files:
                connected_components += [json.load(open(f))]
                progress.update(load_task, advance =1)
    
    return connected_components
