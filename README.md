# Blockchain Interoperability


This repository contains code related to social media data on blockchain interoperability (specifically hack risks)

Requirements for running all scripts can be found in `requirements.txt`. (Python 3.6)
In the local setup, the virtual environment is located in `blockchain-sns-env/`, one level below the root directory
The environment can be activated by running `source blockchain-sns-env/bin/activate`. 

The requirements file should be updated by running `pip freeze > requirements.txt` **while you are in in the virtual environment**

## Directory Structure
```bash
blockchain-social-media
├─analysis
└─twitter-nlp

```

- twitter-nlp
    - Contains code for collecting 
- analysis
    - twitter
        - Contains code for analyzing collected results from twitter
    - utils
        - Dataset definition, common mechanisms etc should be stored here and used as dependencies in other analysis code

... And more social media to be added!


#### Instructions for running twitter collection
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
