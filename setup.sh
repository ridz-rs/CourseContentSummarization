#!/bin/bash
pip install virtualenv
python -m venv summ-env
source summ-env/bin/activate
#env/Scripts/Activate.ps1 //In Powershel
pip install -r requirements.txt
python download_nltk_pakages.py
wget https://storage.googleapis.com/phrase-bert/phrase-bert/phrase-bert-model.zip
unzip phrase-bert-model.zip -d phrase-bert-model/
rm phrase-bert-model.zip