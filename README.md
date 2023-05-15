# Blockchain Interoperability

This repository contains the data and code used for our paper titled "Deciphering Crypto Social Media: Analyzing Crypto Twitter to Detect Risk Signals".

Requirements for running all scripts can be found in `requirements.txt`. (Python 3.9)
In the local setup, the virtual environment is located in `/data/blockchain-interoperability/blockchain-social-media/blockchain-sns-env/`, one level below the root directory
The environment can be activated by running `source /data/blockchain-interoperability/blockchain-social-media/blockchain-sns-env/bin/activate`. 

The requirements file should be updated by running `pip freeze > requirements.txt` **while you are in in the virtual environment**

**Tips and tricks for using jupyter on the browser**
To use the virtual environment on the jupyter kernel, run the following commands while **inside** the virtual environment


Once successful, run this command **after** you enter the virtual environment.
```bash
pip install ipykernel 
python -m ipykernel install --user --name=blockchain-sns-env
```

Now the `blockchain-sns-env` environment should appear as a choice in the jupyter interface.

If the environment needs to be removed for any reason, run the following command
```
jupyter kernelspec remove blockchain-sns-env
```

## Converting the tweets to embeddings

The `main.py` can be used to cache the elasticsearch index into python pickles of dataframes which can be easily accessed by python code.

The settings for `main.py` are stored in `config.json`.


## How to use the streamlit dashboard

Credits to Abraham Sanders to the [original code](https://github.com/TheRensselaerIDEA/twitter-nlp)

To run the dashboard on your local machine, first ensure that elasticsearch is exposed through port 8080.
The dashboard can then be accessed by running 

```streamlit run demo_streamlit.py```

inside the `tools/aspect_modeling` directory.


## Starting Twitter collection for social media

Easiest way: Use the terminal in jupyter kernel for the idea cluster. (You **need** to keep track of the process that spawned the collection. So using the jupyter terminal is encouraged because the sessions persist, unlike ssh)
1. Actvate venv (blockchain-env)
2. navigate to twitter folder by `cd twitter-nlp`
3. Run the collection process by `bash run_twitter.sh`

The configuration file can be found under `twitter-nlp/twitter-monitor/blockchain_config.json`.


## Collection Period

Start date: `Timestamp('2022-11-09 05:53:33.788000')`

End date: `Timestamp('2022-11-23 20:58:16.667000')`

use `pd.to_datetime()` to convert strings to datetime objects for consistency

All the heavy operations should be ran through main.py. Only once the cache is created locally should you use the data on jupyter.


## Acknowledgements

The authors acknowledge the support from NSF IUCRC CRAFT center research grant (CRAFT Grant #22006) for this research. The opinions expressed in this code base do not necessarily represent the views of NSF IUCRC CRAFT.
