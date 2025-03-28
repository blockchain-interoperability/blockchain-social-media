from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
import time
from rich.progress import Progress

from crypto_chatter.config import CryptoChatterDataConfig

from .prettify_elastic_twitter import prettify_elastic_twitter

def prettify_elastic(
    results:list[dict],
    data_config:CryptoChatterDataConfig,
    progress: Progress|None = None,
) -> pd.DataFrame:
    if data_config.data_source == "twitter":
        return prettify_elastic_twitter(
            results=results, 
            data_config=data_config,
            progress=progress
        )
    elif data_config.data_source == "reddit": 
        raise NotImplementedError("Reddit parsing is not yet implemented!")

def load_snapshots(
    data_config: CryptoChatterDataConfig,
    progress: Progress|None = None,
) -> pd.DataFrame:
    """Grabs the specified fields from the specified index on Elasticsearch. Since results are expected to be larged, batched pickle files are generated

    Args:
        SNAPSHOT_DIR (str, optional): Path of directory that stores snapshots. Defaults to "".
    Returns:
        df (pd.DataFrame): A concatenated dataframe containing all the specified columns.
    """
    data_config.raw_snapshot_dir.mkdir(exist_ok=True, parents=True)
    marker_file = data_config.raw_snapshot_dir / "completed.txt"

    if not marker_file.is_file():
        chunk_size = 100000
        es = Elasticsearch(
            hosts=[data_config.es_hostname],
            verify_certs=False,
        )

        # we will add a query to only grab the ones that contain at least one keyword (or partially, if keyword was space separated)
        doc_count = es.count(
            index=[data_config.es_index],
            body=data_config.es_query,
            request_timeout = 120,
        )["count"]
        print(f"scanning {doc_count:,} documents..")

        cursor = scan(
            es,
            index=data_config.es_index,
            query = {**data_config.es_query, "_source": data_config.es_columns},
            request_timeout = 120,
            scroll="1h",
        )

        dataframes = []
        results = []
        start = time.time()

        if progress is not None:
            progress_task = progress.add_task(
                description="scrolling index..", 
                total=doc_count
            )

        for c in cursor:
            results += [c]
            if len(results) == chunk_size:
                df = prettify_elastic(
                    results=results, 
                    data_config=data_config,
                    progress=progress,
                )
                df.to_pickle(
                    data_config.raw_snapshot_dir / f"{len(dataframes):010d}.pkl", 
                )
                del results[:]
                dataframes += [df]
            if progress is not None:
                progress.update(progress_task, advance = 1)


        df = prettify_elastic(
            results=results, 
            data_config=data_config,
            progress=progress,
        )
        if progress is not None:
            progress.remove_task(progress_task)
        del results[:]
        df.to_pickle(
            data_config.raw_snapshot_dir / f"{len(dataframes):010d}.pkl", 
        )
        dataframes += [df]

        num_rows = (len(dataframes) -1) * chunk_size + len(results)

        print(f"we saved {num_rows:,} rows in {len(dataframes)} chunks in {time.time()-start:.2f} seconds")
        df = pd.concat(dataframes).reset_index(drop=True)
        marker_file.touch()

    else:
        start = time.time()
        dataframes = []
        cache_files = sorted(data_config.raw_snapshot_dir.glob("*.pkl"))

        progress_task = None
        if progress is not None:
            progress_task = progress.add_task(
                description="loading snapshots from cache..", 
                total=len(cache_files)
            )

        for file in cache_files:
            dataframes += [pd.read_pickle(file)]
            if progress is not None:
                progress.update(progress_task, advance=1)
        if progress is not None:
            progress.remove_task(progress_task)

        df = pd.concat(dataframes).reset_index(drop=True)
        print(f"Loaded snapshot cache [{len(df)}] rows in {time.time()-start:.2f} seconds")

    return df
