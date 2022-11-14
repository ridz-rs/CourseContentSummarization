#!/bin/bash
pip install virtualenv
python -m venv summ-env
source summ-env/bin/activate
#env/Scripts/Activate.ps1 //In Powershel
pip install -r requirements.txt
python download_nltk_pakages.py
