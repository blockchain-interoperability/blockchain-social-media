# Blockchain Interoperability

This repository contains the data and code used for our paper titled "Decrypting Crypto Social Media: Analyzing Crypto Twitter to Detect Risk Signals".

The `twitter` folder contains the python code for running our analysis. the specific plot generation code can be found in the jupyter notebooks under `twitter/notebooks`.

## Running the code
Requirements for running all scripts can be found in `requirements.txt`. (Python 3.9)
In the local setup, the virtual environment is located in `/data/blockchain-interoperability/blockchain-social-media/blockchain-sns-env/`, one level below the root directory
The environment can be activated by running `source /data/blockchain-interoperability/blockchain-social-media/blockchain-sns-env/bin/activate`. 

The requirements file should be updated by running `pip freeze > requirements.txt` **while you are in in the virtual environment**

## How to use the streamlit dashboard

Credits to Abraham Sanders to the [original code](https://github.com/TheRensselaerIDEA/twitter-nlp)

To run the dashboard on your local machine, first ensure that elasticsearch is exposed through port 8080.
The dashboard can then be accessed by running 

```streamlit run demo_streamlit.py```

inside the `tools/aspect_modeling` directory.


## Starting collection for social media

### Instructions for running twitter collection
Easiest way: Use the terminal in jupyter kernel for the idea cluster. (You **need** to keep track of the process that spawned the collection. So using the jupyter terminal is encouraged because the sessions persist, unlike ssh)
1. Actvate venv (blockchain-env)
2. navigate to twitter folder by `cd twitter-nlp`
3. Run the collection process by `bash run_twitter.sh`

The configuration file can be found under `twitter-nlp/twitter-monitor/blockchain_config.json`.

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

### Instructions for running steemit collection

`cd` into the steemit folder and run `python3 run_steemit.py`. 
The results are stored in the `steemit-data.csv` file by default.
The progress can be tracked in the `steemit.log` file

## Analyzing Social media data

This project uses Spacy to parse the text values. Ensure that the language model is downloaded before proceeding this part by running `python3 -m spacy download en_core_web_lg` while in venv.

### Twitter data


Start date: `Timestamp('2022-11-09 05:53:33.788000')`

End date: `Timestamp('2022-11-23 20:58:16.667000')`


use `pd.to_datetime()` to convert strings to datetime objects for consistency


## Acknowledgements

The authors acknowledge the support from NSF IUCRC CRAFT center research grant (CRAFT Grant #22006) for this research. The opinions expressed in this code base do not necessarily represent the views of NSF IUCRC CRAFT.
