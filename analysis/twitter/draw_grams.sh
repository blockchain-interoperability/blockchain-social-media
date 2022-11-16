# bash

source /data/blockchain-interoperability/blockchain-social-media/blockchain-sns-env/bin/activate
cd /data/blockchain-interoperability/blockchain-social-media/analysis/twitter
python3 main.py draw_ngrams_mixed
python3 main.py draw_ngrams_emoji
python3 main.py draw_ngrams_text