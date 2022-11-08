import json
import argparse
from time import sleep
import logging
from pathlib import Path
import pandas as pd
from steem import Steem

# argument setup
parser = argparse.ArgumentParser()
parser.add_argument('--config_file',default="config.json", required=False, help="Path to the config file to use.")
parser.add_argument('--log_file',default="steemit.log", required=False, help="Path to the logging file to use.")
args = parser.parse_args()

# config loaded
config = json.load(open(args.config_file))

# logging setup
logging.basicConfig(filename=args.log_file, 
                        format="[%(asctime)s - %(levelname)s]: %(message)s", 
                        level=logging.getLevelName(config['log_level']))

# load prev collected
output_file = Path(config['output_file'])
if output_file.is_file(): collected = pd.read_csv(output_file)
else: collected = pd.DataFrame(columns=config['data_columns'])

# collection time!
stm = Steem(nodes=[config['nodename']])

while True:
    try:
        posts = stm.get_discussions_by_created(config['query'])
    except:
        logging.error('request failed.. we sleep for 30s and try again')
        sleep(30)
        pass
    new_posts = pd.DataFrame(posts)
    new_posts = new_posts[
        # shouldn't be repeated
        ~new_posts.post_id.isin(collected.post_id) 
        & (
            # collected must b empty or the date must not be before
            (len(collected) == 0)
            | (pd.to_datetime(new_posts.created) > pd.to_datetime(collected.created).max())
        )    
    ]
    
    new_posts.to_csv(
        output_file,
        index=False,
        header=False if output_file.is_file() else True,
        mode = 'a' if output_file.is_file() else 'w'
    )

    collected = pd.concat([collected,new_posts])

    logging.info(f'we have {len(new_posts)} posts')

    if len(new_posts) == 0:
        logging.info(f"we sleep for {config['wait_time']} secs")
        sleep(config['wait_time'])  