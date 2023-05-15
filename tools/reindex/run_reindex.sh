# !/bin/bash

echo $1

source /data/blockchain-interoperability/blockchain-social-media/blockchain-sns-env/bin/activate
cd ~/blockchain-social-media/twitter/tools/reindex

for config in "$1/"*".json";

do python3 reindex.py --sourceindex blockchain-interoperability-attacks --configfile $config
# do echo $config
done
deactivate